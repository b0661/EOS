import json
import tempfile
from pathlib import Path

import pytest

from akkudoktoreos.config import configmigrate
from akkudoktoreos.config.config import ConfigEOS, SettingsEOSDefaults
from akkudoktoreos.core.version import __version__


class TestConfigMigration:
    """Tests for migrate_config_file()"""

    @pytest.fixture
    def tmp_config_file(self, config_default_dirs) -> Path:
        """Create a temporary valid config file with an invalid version."""
        config_default_dir_user, _, _, _ = config_default_dirs
        config_file_user = config_default_dir_user.joinpath(ConfigEOS.CONFIG_FILE_NAME)

        # Create a default config object (simulates the latest schema)
        default_config = SettingsEOSDefaults()

        # Dump to JSON
        config_json = json.loads(default_config.model_dump_json())

        # Corrupt the version (simulate outdated config)
        config_json["general"]["version"] = "0.0.0-old"

        # Write file
        with config_file_user.open("w", encoding="utf-8") as f:
            json.dump(config_json, f, indent=4)

        return config_file_user

    @pytest.fixture
    def tmp_old_config_file(self, config_default_dirs) -> Path:
        """Create a temporary config file simulating an outdated configuration."""
        config_default_dir_user, _, _, _ = config_default_dirs
        config_file_user = config_default_dir_user.joinpath(ConfigEOS.CONFIG_FILE_NAME)

        old_config = {
            "general": {
                "version": "0.0.0-invalid",  # old version, triggers migration
                "latitude": 50.0,
                "longitude": 8.0,
            },
            "logging": {
                "level": "DEBUG",  # should be migrated → logging/console_level
            },
        }

        with config_file_user.open("w", encoding="utf-8") as f:
            json.dump(old_config, f, indent=4)
        return config_file_user

    def test_migrate_config_file_from_invalid_version(self, tmp_config_file: Path):
        """Test that migration updates an outdated config version successfully."""
        backup_file = tmp_config_file.with_suffix(".bak")

        # Run migration
        result = configmigrate.migrate_config_file(tmp_config_file, backup_file)

        # Verify success
        assert result is True, "Migration should succeed even from invalid version."

        # Verify backup exists
        assert backup_file.exists(), "Backup file should be created before migration."

        # Verify version updated
        with tmp_config_file.open("r", encoding="utf-8") as f:
            migrated_data = json.load(f)
        assert migrated_data["general"]["version"] == __version__, \
            "Migrated config should have updated version."

        # Verify it still matches the structure of SettingsEOSDefaults
        new_model = SettingsEOSDefaults(**migrated_data)
        assert isinstance(new_model, SettingsEOSDefaults)

    def test_migrate_config_file_already_current(self, tmp_path: Path):
        """Test that a current config file returns True immediately."""
        config_path = tmp_path / "EOS_current.json"
        default = SettingsEOSDefaults()
        with config_path.open("w", encoding="utf-8") as f:
            f.write(default.model_dump_json(indent=4))

        backup_file = config_path.with_suffix(".bak")

        result = configmigrate.migrate_config_file(config_path, backup_file)
        assert result is True
        assert not backup_file.exists(), "No backup should be made if config is already current."


    def test_migrate_old_version_config(self, tmp_old_config_file: Path):
        """Migration should succeed and rewrite the configuration."""
        backup_file = tmp_old_config_file.with_suffix(".bak")

        # Perform migration
        result = configmigrate.migrate_config_file(tmp_old_config_file, backup_file)

        # --- Assertions ---
        # 1. Migration must succeed
        assert result is True, "Migration should complete successfully"

        # 2. Migration statistics must indicate actual work
        assert configmigrate.mapped_count >= 1, "Expected at least one mapped migration"
        assert configmigrate.auto_count >= 1, "Expected some automatic migrations"

        # 3. Skipped paths must not be excessive
        # (small configs may skip a few optional fields)
        assert len(configmigrate.skipped_paths) <= 1, (
            f"Too many fields skipped: {configmigrate.skipped_paths}"
        )

        # 4. Migrated paths should include known mappings
        assert "logging/level" in configmigrate.migrated_source_paths, \
            "Expected migration of logging/level"

        # 5. New file should contain the updated version and migrated fields
        with tmp_old_config_file.open("r", encoding="utf-8") as f:
            new_data = json.load(f)

        assert new_data["general"]["version"] == __version__, \
            "Migrated config should include the current version"
        assert "console_level" in new_data["logging"], \
            "Expected migrated logging/console_level field"
