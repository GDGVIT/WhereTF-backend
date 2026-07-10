import os
import shutil
import zipfile
import logging

from app.database import SessionLocal
from app.models import File, FileContent
from app.processing import process
from app.services.archive import extract_zip

logger = logging.getLogger(__name__)


def save_processed_data(db, file_rows, content_rows):
    """
    Save processed file metadata and chunks into the database.
    """

    for fr in file_rows:
        existing = db.query(File).filter_by(file_path=fr["file_path"]).first()

        if existing:
            for k, v in fr.items():
                setattr(existing, k, v)
        else:
            db.add(File(**fr))

    db.flush()

    for cr in content_rows:

        file_path = cr.pop("file_path")

        file_obj = (
            db.query(File)
            .filter_by(file_path=file_path)
            .one()
        )

        db.add(
            FileContent(
                file_id=file_obj.id,
                chunk_index=cr["chunk_index"],
                content_text=cr["content_text"],
                embedding=cr["embedding"],
            )
        )


def background_index_file(temp_file_path: str, original_path: str):
    """
    Processes a document (or ZIP archive), generates embeddings,
    stores everything in PostgreSQL, and cleans up temporary files.
    """

    db = SessionLocal()

    try:

        logger.info(f"Starting indexing for: {temp_file_path}")

        # --------------------------------------------------------
        # ZIP FILE
        # --------------------------------------------------------

        if zipfile.is_zipfile(temp_file_path):

            logger.info("ZIP archive detected.")

            temp_extract_dir, extracted_files = extract_zip(
                temp_file_path
            )

            try:

                for extracted in extracted_files:

                    extension = os.path.splitext(extracted)[1].lower()

                    supported = {
                        ".pdf",
                        ".docx",
                        ".txt",
                        ".md",
                        ".pptx",
                        ".csv",
                        ".py",
                    }

                    if extension not in supported:
                        logger.info(f"Skipping unsupported file: {extracted}")
                        continue

                    logger.info(f"Processing: {extracted}")

                    archive_path = (
                        f"{original_path}::"
                        f"{os.path.relpath(extracted, temp_extract_dir)}"
                    )

                    file_rows, content_rows = process(extracted)

                    for fr in file_rows:
                        fr["file_path"] = archive_path

                    for cr in content_rows:
                        cr["file_path"] = archive_path

                    save_processed_data(
                        db,
                        file_rows,
                        content_rows
                    )

                db.commit()

                logger.info("ZIP archive indexed successfully.")

            finally:

                shutil.rmtree(temp_extract_dir)

            return

        # --------------------------------------------------------
        # NORMAL FILE
        # --------------------------------------------------------

        file_rows, content_rows = process(temp_file_path)

        for fr in file_rows:
            fr["file_path"] = original_path

        for cr in content_rows:
            cr["file_path"] = original_path

        save_processed_data(
            db,
            file_rows,
            content_rows
        )

        db.commit()

        logger.info(
            f"Successfully indexed "
            f"{len(file_rows)} file(s) "
            f"and {len(content_rows)} chunk(s)."
        )

    except Exception as e:

        db.rollback()

        logger.error(
            f"Failed to index {temp_file_path}: {e}"
        )

    finally:

        db.close()

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

            logger.info(
                f"Removed temporary file: {temp_file_path}"
            )