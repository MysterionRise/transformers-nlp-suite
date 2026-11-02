"""Unit tests for launch_ui.py"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from launch_ui import UIS, launch_ui, list_uis, print_banner  # noqa: E402


class TestUIConfiguration:
    """Test UI configuration"""

    def test_uis_dict_structure(self):
        """Test that UIS dictionary has the expected structure"""
        assert isinstance(UIS, dict)
        assert len(UIS) > 0

        for key, value in UIS.items():
            assert isinstance(key, str)
            assert isinstance(value, dict)
            assert "name" in value
            assert "file" in value
            assert "port" in value
            assert "description" in value

    def test_all_ui_files_exist(self):
        """Test that all UI files referenced in UIS exist"""
        project_root = Path(__file__).parent.parent.parent
        for ui_info in UIS.values():
            ui_file = project_root / ui_info["file"]
            assert ui_file.exists(), f"UI file {ui_file} does not exist"

    def test_unique_ports(self):
        """Test that all UIs have unique ports"""
        ports = [ui_info["port"] for ui_info in UIS.values()]
        assert len(ports) == len(set(ports)), "Duplicate ports found"

    def test_expected_uis(self):
        """Test that expected UIs are present"""
        expected_keys = ["sentiment", "similarity", "ner", "summarization"]
        for key in expected_keys:
            assert key in UIS, f"Expected UI '{key}' not found"


class TestListUIs:
    """Test list_uis function"""

    def test_list_uis_runs_without_error(self, capsys):
        """Test that list_uis prints output"""
        list_uis()
        captured = capsys.readouterr()
        assert "Available UIs:" in captured.out
        assert "sentiment" in captured.out.lower()


class TestPrintBanner:
    """Test print_banner function"""

    def test_print_banner_runs_without_error(self, capsys):
        """Test that print_banner prints output"""
        print_banner()
        captured = capsys.readouterr()
        assert "NLP" in captured.out
        assert "=" in captured.out


class TestLaunchUI:
    """Test launch_ui function"""

    def test_launch_ui_invalid_key(self, capsys):
        """Test launching with invalid UI key"""
        result = launch_ui("invalid_ui_key")
        assert result is False
        captured = capsys.readouterr()
        assert "Error" in captured.out

    @patch("subprocess.run")
    def test_launch_ui_valid_key(self, mock_run):
        """Test launching with valid UI key"""
        mock_run.return_value = Mock()
        launch_ui("sentiment")
        assert mock_run.called

    @patch("subprocess.run")
    def test_launch_ui_keyboard_interrupt(self, mock_run, capsys):
        """Test handling keyboard interrupt"""
        mock_run.side_effect = KeyboardInterrupt()
        result = launch_ui("sentiment")
        assert result is True
        captured = capsys.readouterr()
        assert "Shutting down" in captured.out

    @patch("subprocess.run")
    def test_launch_ui_exception(self, mock_run, capsys):
        """Test handling general exception"""
        mock_run.side_effect = Exception("Test error")
        result = launch_ui("sentiment")
        assert result is False
        captured = capsys.readouterr()
        assert "Error launching UI" in captured.out


class TestCatalogAndAllInOne:
    """The launcher derives its list from ui/catalog.py and can start the combined app."""

    def test_uis_match_catalog(self):
        from ui.catalog import UIS as CATALOG

        assert list(UIS) == list(CATALOG)
        for key, entry in CATALOG.items():
            assert UIS[key]["port"] == entry.port
            assert UIS[key]["module"] == f"ui.{entry.module}"

    @patch("subprocess.run")
    def test_launch_runs_module_from_repo_root(self, mock_run):
        import launch_ui as launcher

        assert launch_ui("qa") is True
        args, kwargs = mock_run.call_args
        assert args[0][1:] == ["-m", "ui.qa_system"]
        assert kwargs["cwd"] == launcher.PROJECT_ROOT

    @patch("subprocess.run")
    def test_launch_all_forwards_host_and_port(self, mock_run):
        assert launch_ui("all", host="0.0.0.0", port=9000) is True
        assert mock_run.call_args.args[0][1:] == ["-m", "ui.app", "--host", "0.0.0.0", "--port", "9000"]
