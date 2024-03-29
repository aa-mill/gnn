import regular
import irregular
import argparse

def newDataset(type, train_size, val_size, test_size, dir):
    """
    Creates new train, validation, and test datasets.

    Parameters:
        train_size (int): Number of training samples.
        val_size (int): Number of validation samples.
        test_size (int): Number of test samples.

    Returns:
        None
    """
    if type == 'line':
        regular.createData(train_size, 
                           dim=1, 
                           output_path=f'{dir}train_data.pkl', 
                           nps=2)
        regular.createData(val_size, 
                           dim=1, 
                           output_path=f'{dir}val_data.pkl',
                           nps=2)
        regular.createData(test_size, 
                           dim=1, 
                           output_path=f'{dir}test_data.pkl', 
                           raw=True, 
                           nps=2)
    elif type == 'block':
        regular.createData(train_size, 
                           dim=2, 
                           output_path=f'{dir}train_data.pkl')
        regular.createData(val_size, 
                           dim=2, 
                           output_path=f'{dir}val_data.pkl')
        regular.createData(test_size, 
                           dim=2, 
                           output_path=f'{dir}test_data.pkl', 
                           raw=True)
    elif type == 'irregular':
        irregular.createData(train_size, 
                             output_path=f'{dir}train_data.pkl')
        irregular.createData(val_size, 
                             output_path=f'{dir}val_data.pkl')
        irregular.createData(test_size, 
                             output_path=f'{dir}test_data.pkl', 
                             raw=True)

def parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Generate train, validation, and test sets.')
    parser.add_argument('--type', type=str, required=True,
                        help='Type of dataset to generate.')
    parser.add_argument('--train_size', type=int, default=1000, 
                        help='Number of training samples.')
    parser.add_argument('--val_size', type=int, default=100, 
                        help='Number of validation samples.')
    parser.add_argument('--test_size', type=int, default=100, 
                        help='Number of test samples.')
    parser.add_argument('--dir', type=str, default='data/',
                        help='Directory to save datasets.')
    return parser.parse_args()

def main():
    args = parse_args()
    newDataset(args.type, 
               args.train_size, 
               args.val_size, 
               args.test_size,
               args.dir)

if __name__ == '__main__':
    main()