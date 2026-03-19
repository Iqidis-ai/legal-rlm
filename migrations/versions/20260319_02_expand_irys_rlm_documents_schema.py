"""Expand irys_rlm_documents to the virtual-document schema."""

from alembic import op
import sqlalchemy as sa

revision = "20260319_02"
down_revision = "20260319_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "irys_rlm_documents",
        "url",
        existing_type=sa.String(length=2048),
        existing_nullable=False,
        new_column_name="canonical_url",
    )
    op.alter_column(
        "irys_rlm_documents",
        "extracted_text",
        existing_type=sa.Text(),
        existing_nullable=True,
        new_column_name="full_text",
    )
    op.add_column(
        "irys_rlm_documents",
        sa.Column("source_url", sa.String(length=2048), nullable=True),
    )
    op.add_column(
        "irys_rlm_documents",
        sa.Column("byte_size", sa.Integer(), nullable=True),
    )
    op.add_column(
        "irys_rlm_documents",
        sa.Column("page_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "irys_rlm_documents",
        sa.Column("total_chars", sa.Integer(), nullable=True),
    )
    op.add_column(
        "irys_rlm_documents",
        sa.Column("pages_json", sa.JSON(), nullable=True),
    )
    op.add_column(
        "irys_rlm_documents",
        sa.Column("extraction_status", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "irys_rlm_documents",
        sa.Column("extraction_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "irys_rlm_documents",
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "irys_rlm_documents",
        sa.Column("fetch_verified_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        "UPDATE irys_rlm_documents SET source_url = canonical_url WHERE source_url IS NULL"
    )
    op.alter_column(
        "irys_rlm_documents",
        "source_url",
        existing_type=sa.String(length=2048),
        nullable=False,
    )
    op.create_index(
        "ix_irys_rlm_documents_extracted_at",
        "irys_rlm_documents",
        ["extracted_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_irys_rlm_documents_extracted_at", table_name="irys_rlm_documents")
    op.drop_column("irys_rlm_documents", "fetch_verified_at")
    op.drop_column("irys_rlm_documents", "extracted_at")
    op.drop_column("irys_rlm_documents", "extraction_version")
    op.drop_column("irys_rlm_documents", "extraction_status")
    op.drop_column("irys_rlm_documents", "pages_json")
    op.drop_column("irys_rlm_documents", "total_chars")
    op.drop_column("irys_rlm_documents", "page_count")
    op.drop_column("irys_rlm_documents", "byte_size")
    op.drop_column("irys_rlm_documents", "source_url")
    op.alter_column(
        "irys_rlm_documents",
        "full_text",
        existing_type=sa.Text(),
        existing_nullable=True,
        new_column_name="extracted_text",
    )
    op.alter_column(
        "irys_rlm_documents",
        "canonical_url",
        existing_type=sa.String(length=2048),
        existing_nullable=False,
        new_column_name="url",
    )