"""Create irys_rlm_documents table."""

from alembic import op
import sqlalchemy as sa

revision = "20260319_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "irys_rlm_documents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("url", sa.String(length=2048), nullable=False),
        sa.Column("file_name", sa.String(length=512), nullable=True),
        sa.Column("content_type", sa.String(length=255), nullable=True),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("checksum", sa.String(length=128), nullable=True),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("url"),
    )
    op.create_index(
        "ix_irys_rlm_documents_updated_at",
        "irys_rlm_documents",
        ["updated_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_irys_rlm_documents_updated_at", table_name="irys_rlm_documents")
    op.drop_table("irys_rlm_documents")