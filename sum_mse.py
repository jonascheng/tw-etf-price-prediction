import re
import sys


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print('Please specify file to be calculated')
        exit()

    filename = sys.argv[1]

    sum_mse = 0.0
    with open(filename, 'r') as file:
        lines = file.readlines()
        assert len(lines)==18
        for line in lines:
            tokens = re.split(' |\[|\]', line.strip())
            print('{} {}'.format(tokens[0], float(tokens[2])))
            sum_mse += float(tokens[2])

    print('Sum MSE: {}'.format(sum_mse))


