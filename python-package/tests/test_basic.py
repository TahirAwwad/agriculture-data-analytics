from project_name.project_manager import ProjectArtifactManager, ProjectAssetManager
import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_ProjectAssetManager(self):
        asset_manager: ProjectAssetManager = ProjectAssetManager()
        actual: str = "./../assets/2021-12Dec-11-population-estimates-1950-2021-pea01.csv"
        self.assertEqual(
            asset_manager.get_population_estimates_filepath(), actual)

    def test_ProjectArtifactManager(self):
        artifact_manager: ProjectArtifactManager = ProjectArtifactManager()
        actual: str = "./../artifacts/population-1950-2021-eda-output.csv"
        self.assertEqual(
            artifact_manager.get_population_eda_filepath(), actual)


if __name__ == '__main__':
    unittest.main()
