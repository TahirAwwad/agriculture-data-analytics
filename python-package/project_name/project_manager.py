# Copyright (c) 2021 Mark Crowe <https://github.com/markcrowe-com>. All rights reserved.

from data_analytics.github import RepositoryFileManager, RELATIVE_PATH


REPOSITORY_URL = 'https://github.com/markcrowe-com/data-analytics-project-template'


class ProjectArtifactManager(RepositoryFileManager):

    def __init__(self, relative_path: str = RELATIVE_PATH, repository_url: str = REPOSITORY_URL, is_remote: bool = False):
        super().__init__(repository_url, relative_path, is_remote)
        self.POPULATION_EDA_FILENAME: str = "artifacts/population-1950-2021-eda-output.csv"

    def get_population_eda_filepath(self) -> str:
        return super().get_repository_filepath(self.POPULATION_EDA_FILENAME)


class ProjectAssetManager(RepositoryFileManager):

    def __init__(self, relative_path: str = RELATIVE_PATH, repository_url: str = REPOSITORY_URL, is_remote: bool = False):
        super().__init__(repository_url, relative_path, is_remote)
        self.POPULATION_ESTIMATES_FILENAME: str = "assets/2021-12Dec-11-population-estimates-1950-2021-pea01.csv"

    def get_population_estimates_filepath(self) -> str:
        return super().get_repository_filepath(self.POPULATION_ESTIMATES_FILENAME)
