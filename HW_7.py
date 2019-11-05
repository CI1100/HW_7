import argparse
import os.path as op
import csv
import matplotlib.pyplot as plt
import numpy as np

def convert_type(data_value):
    try:
        return int(data_value)
    except ValueError:
        try:
            return float(data_value)
        except ValueError:
            return data_value

def lines_to_dict(lines, header=False):
    if header:
        column_titles = lines[0]
        lines = lines[1:]
    else:
        column_titles = list(range(1, len(lines[0])+1))
    
    data_dict = {}
    for idx, column in enumerate(column_titles):
        data_dict[column] = []
        for row in lines:
            data_dict[column] += [row[idx]]
    return data_dict

def parse_file(data_file, delimiter, debug=False):
    # Verify the file exists
    print('Parsing data file: ', data_file)
    assert(op.isfile(data_file))

    # open it as a csv (not checking delimiters, so you can do better)
    with open(data_file, 'r') as fhandle:
        csv_reader = csv.reader(fhandle, delimiter=delimiter)

        # Add each line in the file to a list
        lines = []
        if debug:
            count = 0
        for line in csv_reader:
            if debug:
                if count > 2:
                    break
                count += 1
            newline = []
            for value in line:
                newline += [convert_type(value)]

            if len(newline) > 0:
                lines += [newline]

    # Return all the contents of our file
    return lines

def generate_points(coefs, min_val, max_val):
    xs = np.arange(min_val, max_val, (max_val-min_val)/100)
    return xs, np.polyval(coefs, xs)

def plot_data(dd, debug=False, polys=[1,2,3,4]):
    # dd stands for data_dictionary, debug doesn't plot
    if debug:
        number_combinations = 0

    ncols = len(dd.keys())
    if not debug:
        fig = plt.Figure(figsize=(30, 30))   
    for i1, column1 in enumerate(dd.keys()):
        for i2, column2 in enumerate(dd.keys()):
            if debug:
                number_combinations += 1
                print(column1, column2)
                # import pdb
                # pdb.set_trace()
            else:
                # If my grid is :
                # 1  2  3  4  5
                # 6  7  8  9  10
                # 11 12 13 14 15
                #  ... then, I want to index it at i1*ncols + i2   (+1)
                loc = i1*ncols + i2 + 1
                plt.subplot(ncols, ncols, loc)
                x = dd[column1]
                y = dd[column2]
                
                plt.scatter(x, y)
                plt.xlabel(column1)
                plt.ylabel(column2)
                plt.title("{0} x {1}".format(column1, column2))

                for poly_order in polys:
                    coefs = np.polyfit(x, y, poly_order)  # we also want to do this for 2, 3
                    xs, new_line = generate_points(coefs, min(x), max(x))
                    plt.plot(xs, new_line)
    if not debug:
        # Note: I have spent no effort making it pretty, and recommend that you do :)
        plt.legend()
        # plt.tight_layout()
        # plt.show()
        plt.savefig("./my_pairs_plot.png")

    if debug:
        print(len(dd.keys()), number_combinations)
    return 0


def calculate_summary(data_dictionary, column, debug=False):
    
     if column not in data_dictionary:
        print('Invalid column name: ', column)
        return
     
     data = data_dictionary[column]
    
     data_type = 'categorical'
     array = np.array(data)
     if array.dtype == np.float:
        data_type = 'continuous'
     elif array.dtype == np.int:
        data_type = 'discrete'
        
     if data_type != 'categorical':
        max = np.max(array)
        min = np.min(array)
        stdev = np.std(array)
        mean = np.mean(array)
     
     print('=== Summary for "%s" ==='.format(column))
     print('Guessed data type: ', data_type)
     
     if data_type != 'categorical':
        print('Min: ', min)
        print('Max: ', max)
        print('Stdev: ', stdev)
        print('Mean: ', mean)
     
def calculate_interpolation(data_dictionary, interpolation):

    column1, column2, value = interpolation.split(",")
    print('Params "%s", "%s", "%s"'.format(column1, column2, value))
    value = float(value)
    column1 = column1.strip()
    column2 = column2.strip()   
    
    if column1 not in data_dictionary:
        print('Invalid column name: ', column1)
        return
    if column2 not in data_dictionary:
        print('Invalid column name: ', column2)
        return        
    max_value = np.max(data_dictionary[column1])
    min_value = np.min(data_dictionary[column1])

    x = np.array(data_dictionary[column1])
    y = np.array(data_dictionary[column2])
    
    data_type = 'categorical'
    if y.dtype == np.float:
       data_type = 'continuous'
    elif y.dtype == np.int:
        data_type = 'discrete'
        
    if value > max_value or value < min_value:
        print('Please select value from interval <min, max>, use --summary to find interval')
        return
       
    if data_type == 'categorical':
        print('Cannot interpolate categorical values')
        return

    for poly_order in [1, 2, 3]:
        coefs = np.polyfit(x, y, poly_order)  # we also want to do this for 2, 3
        xs, new_line = generate_points(coefs, min_value, max_value)
        plt.plot(xs, new_line)
        
    plt.legend(['1st order', '2nd order', '3rd order'])
    plt.title("{0} x {1}".format(column1, column2))
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str,
                        help="Input CSV data file for plotting")
    parser.add_argument("delimiter", type=str,
                        help="the delimiter used in your file")
    parser.add_argument('-x', '--debug', action="store_true",
                        help="only prints start of file")
    parser.add_argument('-H', '--header', action="store_true",
                        help="determines if a header is present")
    parser.add_argument('-p', '--plot', help = 'Save plot as mentioned', action='store_true')
    parser.add_argument('-s', '--summary', type=str, help='Specify a column to calculate summray on')
    parser.add_argument('-i', '--interpolation', type=str, help='Specify a pair of columns for interpolation')

    
    args = parser.parse_args()
    my_data = parse_file(args.data_file, args.delimiter, debug=args.debug)
    data_dictionary = lines_to_dict(my_data, header=args.header)
    print(data_dictionary)
    
    if args.plot:
        plot_data(data_dictionary, debug=args.debug)
    if args.summary:
        calculate_summary(data_dictionary, args.summary, debug=args.debug)
    if args.interpolation:
        calculate_interpolation(data_dictionary, args.interpolation)
if __name__ == "__main__":
    main()