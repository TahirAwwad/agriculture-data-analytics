# Copyright (c) 2021 Mark Crowe <https://github.com/markcrowe-com>. All rights reserved.

from data_analytics.github import RepositoryFileManager, RELATIVE_PATH


REPOSITORY_URL = 'https://github.com/markcrowe-com/agriculture-data-analytics'


class ProjectArtifactManager(RepositoryFileManager):

    def __init__(self, relative_path: str = RELATIVE_PATH, repository_url: str = REPOSITORY_URL, is_remote: bool = False):
        super().__init__(repository_url, relative_path, is_remote)
        self.POPULATION_EDA_FILENAME: str = "artifacts/population-1950-2021-eda-output.csv"

    def get_population_eda_filepath(self) -> str:
        return super().get_repository_filepath(self.POPULATION_EDA_FILENAME)


class ProjectAssetManager(RepositoryFileManager):

    def __init__(self, relative_path: str = RELATIVE_PATH, repository_url: str = REPOSITORY_URL, is_remote: bool = False):
        super().__init__(repository_url, relative_path, is_remote)
        self.CATTLE_BEEF_EXPORTS_FILENAME: str = "assets/cso-tsa04-exports-of-cattle-and-beef-1930-2020-2022-01Jan-13.csv"
        self.CATTLE_BEEF_MONTHLY_EXPORTS_FILENAME: str = "assets/cso-tsm04-exports-of-cattle-and-beef-1970-2021-2022-01Jan-13.csv"

    def get_cattle_beef_exports_filepath(self) -> str:
        return super().get_repository_filepath(self.CATTLE_BEEF_EXPORTS_FILENAME)

    def get_cattle_beef_monthly_exports_filepath(self) -> str:
        return super().get_repository_filepath(self.CATTLE_BEEF_MONTHLY_EXPORTS_FILENAME)
