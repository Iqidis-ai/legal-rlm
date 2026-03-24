"""Add user_id column to irys_rlm_investigation_logs.

Adds an optional, indexed user_id column alongside message_id so that
investigations can be attributed to a user even after the originating
message/chat is deleted.  No FK to any external table — this repo owns
its telemetry rows completely.
"""

from alembic import op
import sqlalchemy as sa

revision = "20260324_01"
down_revision = "20260320_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "irys_rlm_investigation_logs",
        sa.Column("user_id", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_irys_rlm_investigation_logs_user_id",
        "irys_rlm_investigation_logs",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_irys_rlm_investigation_logs_user_id",
        table_name="irys_rlm_investigation_logs",
    )
    op.drop_column("irys_rlm_investigation_logs", "user_id")

