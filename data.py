#!/usr/bin/env python3
import regular
import irregular
import argparse

def newDataset(type, nps, train_size, val_size, test_size, dir, double=False):
    """
    Creates new train, validation, and test datasets.

    Parameters:
        type (str): Type of dataset to generate, options are
                    'line', 'block', and 'irregular'.
        nps (int): Neighbors per side, applicable if type is 'line'.
        train_size (int): Number of training samples.
        val_size (int): Number of validation samples.
        test_size (int): Number of test samples.
        dir (str): Directory to save datasets.
        double (bool): If true, uses double precision.

    Returns:
        None
    """
    if type == 'line':
        if nps is None:
            raise ValueError('Number of neighbors must be specified for 1D data.')
        regular.createData(train_size, 
                           dim=1, 
                           output_path=f'{dir}train_data.pkl', 
                           nps=nps,
                           double=double)
        regular.createData(val_size, 
                           dim=1, 
                           output_path=f'{dir}val_data.pkl',
                           nps=nps,
                           double=double)
        regular.createData(test_size, 
                           dim=1, 
                           output_path=f'{dir}test_data.pkl', 
                           raw=True, 
                           nps=nps,
                           double=double)
    elif type == 'block':
        if nps is not None:
            raise ValueError('Number of neighbors is only for 1D data.')
        regular.createData(train_size, 
                           dim=2, 
                           output_path=f'{dir}train_data.pkl',
                           double=double)
        regular.createData(val_size, 
                           dim=2, 
                           output_path=f'{dir}val_data.pkl',
                           double=double)
        regular.createData(test_size, 
                           dim=2, 
                           output_path=f'{dir}test_data.pkl', 
                           raw=True,
                           double=double)
    elif type == 'irregular':
        if nps is not None:
            raise ValueError('Number of neighbors is only for 1D data.')
        irregular.createData(train_size, 
                             output_path=f'{dir}train_data.pkl',
                             double=double)
        irregular.createData(val_size, 
                             output_path=f'{dir}val_data.pkl',
                             double=double)
        irregular.createData(test_size, 
                             output_path=f'{dir}test_data.pkl', 
                             raw=True,
                             double=double)

def parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Generate train, validation, and test sets.')
    parser.add_argument('-t', '--type', type=str, required=True,
                        choices=['line', 'block', 'irregular'],
                        help='Type of dataset to generate.')
    parser.add_argument('--nps', type=int, default=None,
                        help='Neighbors per side, applicable if type is \'line\'.')
    parser.add_argument('--train', type=int, default=1000, 
                        help='Number of training samples.')
    parser.add_argument('--val', type=int, default=100, 
                        help='Number of validation samples.')
    parser.add_argument('--test', type=int, default=100, 
                        help='Number of test samples.')
    parser.add_argument('--dir', type=str, default='data/',
                        help='Directory to save datasets.')
    parser.add_argument('--double', action='store_true',
                        help='If true, uses double precision.')
    return parser.parse_args()

def main():
    args = parse_args()
    newDataset(args.type, 
               args.nps,
               args.train, 
               args.val, 
               args.test,
               args.dir,
               args.double)

if __name__ == '__main__':
    main()