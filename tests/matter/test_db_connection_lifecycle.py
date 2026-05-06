from concurrent.futures import ThreadPoolExecutor

from irys.matter import MatterModel
from irys.matter.db import SQLiteMatterDB


def test_sqlite_matter_db_close_all_closes_worker_thread_handles(tmp_path):
    db_path = tmp_path / "matter.sqlite3"
    db = SQLiteMatterDB(db_path)

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(
            lambda: db.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        ).result()

        db.close_all()

        # The same worker thread must not keep returning its stale closed handle.
        assert pool.submit(
            lambda: db.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        ).result()

        db.close_all()

    for path in (db_path, db_path.with_suffix(".sqlite3-wal"), db_path.with_suffix(".sqlite3-shm")):
        if path.exists():
            path.unlink()


def test_matter_model_close_releases_db_for_reset(tmp_path):
    repo = tmp_path / "matter"
    repo.mkdir()
    model = MatterModel.open(repo)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(
            lambda: model.db.execute("SELECT COUNT(*) FROM matter").fetchone()[0]
        ).result()

        model.close()

    irys_dir = repo / ".irys"
    for path in sorted(irys_dir.iterdir()):
        path.unlink()
    irys_dir.rmdir()
    assert not irys_dir.exists()


def test_clear_matter_closes_registered_handles_before_delete(tmp_path):
    from irys.ui.app import _clear_matter

    repo = tmp_path / "matter"
    repo.mkdir()
    model = MatterModel.open(repo)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(
            lambda: model.db.execute("SELECT COUNT(*) FROM matter").fetchone()[0]
        ).result()

        def close_handles(path):
            assert path.resolve() == repo.resolve()
            model.close()
            return 1

        folder_name, message = _clear_matter(str(repo), close_handles=close_handles)

    assert folder_name == "matter"
    assert "Cleared analysis" in message
    assert not (repo / ".irys").exists()
