import unittest
from pathlib import Path
from ..poll_databases import poll_databases
from ...converters import ds2df
from ...analysis.summary import collect_summary_df, print_pretty_df


PATH_WITH_TESTS = Path(__file__).parent


class yaml_csv_poscars_Test(unittest.TestCase):

    def setUp(self) -> None:
        self.elements = {'Mo', 'Si', 'B', 'P'}

    def test_scrape(self):
        ds = poll_databases(self.elements, do_ehull_filtering=True,
                            loader_kwargs={"alexandria": {'pattern': 'alexandria_00*.json'}})
        print_pretty_df(collect_summary_df(ds), 'table.txt')
        self.assertEqual(len(ds), 65)