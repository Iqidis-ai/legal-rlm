"""Create investigation telemetry tables.

Creates three tables for persisting investigation cost/latency telemetry:
- irys_rlm_investigation_logs (one row per investigation)
- irys_rlm_investigation_steps (one row per step within an investigation)
- irys_rlm_investigation_operations (one row per LLM call or external search)
"""

from alembic import op
import sqlalchemy as sa

revision = "20260320_01"
down_revision = "20260319_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- investigation_logs ---
    op.create_table(
        "irys_rlm_investigation_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("message_id", sa.String(length=255), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("total_duration_ms", sa.Integer(), nullable=True),
        sa.Column("total_cost_usd", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column("total_steps", sa.Integer(), nullable=True),
        sa.Column("phase_breakdown", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_irys_rlm_investigation_logs_message_id",
        "irys_rlm_investigation_logs",
        ["message_id"],
    )
    op.create_index(
        "ix_irys_rlm_investigation_logs_started_at",
        "irys_rlm_investigation_logs",
        ["started_at"],
    )

    # --- investigation_steps ---
    op.create_table(
        "irys_rlm_investigation_steps",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("investigation_id", sa.String(length=36), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("step_name", sa.String(length=128), nullable=False),
        sa.Column("phase", sa.String(length=64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("step_latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["investigation_id"],
            ["irys_rlm_investigation_logs.id"],
        ),
    )
    op.create_index(
        "ix_irys_rlm_investigation_steps_investigation_id",
        "irys_rlm_investigation_steps",
        ["investigation_id"],
    )
    op.create_index(
        "ix_irys_rlm_investigation_steps_step_name",
        "irys_rlm_investigation_steps",
        ["step_name"],
    )

    # --- investigation_operations ---
    op.create_table(
        "irys_rlm_investigation_operations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("step_id", sa.String(length=36), nullable=False),
        sa.Column("investigation_id", sa.String(length=36), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["step_id"],
            ["irys_rlm_investigation_steps.id"],
        ),
    )
    op.create_index(
        "ix_irys_rlm_investigation_operations_step_id",
        "irys_rlm_investigation_operations",
        ["step_id"],
    )
    op.create_index(
        "ix_irys_rlm_investigation_operations_investigation_id",
        "irys_rlm_investigation_operations",
        ["investigation_id"],
    )
    op.create_index(
        "ix_irys_rlm_investigation_operations_type",
        "irys_rlm_investigation_operations",
        ["type"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_irys_rlm_investigation_operations_type",
        table_name="irys_rlm_investigation_operations",
    )
    op.drop_index(
        "ix_irys_rlm_investigation_operations_investigation_id",
        table_name="irys_rlm_investigation_operations",
    )
    op.drop_index(
        "ix_irys_rlm_investigation_operations_step_id",
        table_name="irys_rlm_investigation_operations",
    )
    op.drop_table("irys_rlm_investigation_operations")

    op.drop_index(
        "ix_irys_rlm_investigation_steps_step_name",
        table_name="irys_rlm_investigation_steps",
    )
    op.drop_index(
        "ix_irys_rlm_investigation_steps_investigation_id",
        table_name="irys_rlm_investigation_steps",
    )
    op.drop_table("irys_rlm_investigation_steps")

    op.drop_index(
        "ix_irys_rlm_investigation_logs_started_at",
        table_name="irys_rlm_investigation_logs",
    )
    op.drop_index(
        "ix_irys_rlm_investigation_logs_message_id",
        table_name="irys_rlm_investigation_logs",
    )
    op.drop_table("irys_rlm_investigation_logs")
