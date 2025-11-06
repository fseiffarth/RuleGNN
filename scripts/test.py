from pathlib import Path

from src.Preprocessing.create_splits import splits_from_train_test_files


def main():
    for db in ['MUTAG', 'PTC', 'NCI1', 'PROTEINS', 'IMDBBINARY', 'IMDBMULTI', 'COLLAB',
               'REDDITBINARY', 'REDDITMULTI5K']:
        splits_from_train_test_files(Path(f'tmp/{db}/'), f'{db}')

if __name__ == '__main__':
    main()