import csv
import numpy as np
import statsmodels.api as sm


def load_data(dataset_source):
    with open(dataset_source, 'rU') as infile:
        reader = csv.DictReader(infile)
        data = {}
        for row in reader:
            for header, value in row.items():
                try:
                    data[header].append(value)
                except KeyError:
                    data[header] = [value]
    return data


def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))

    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()

    return results


def print_coefficients(attribute_names, coefficients):
    attribute_names.append("Intercept")
    for i in range(len(attribute_names)):
        print(str(attribute_names[i]) + ": " + str(coefficients[i]))
    attribute_names.remove("Intercept")
    print("\n\n")


def print_suggested(suggested):
    print("Set variable to the suggested value")
    print("to reduce risk to zero.")
    print("************")

    print(suggested)
    print("\n\n")


def get_coefficients(dataset_source, predictor_list, dependent):
    data = load_data(dataset_source)

    data_x = [np.array(data[x]).astype(np.float).tolist() for x in predictor_list]
    data_y = (np.array(data[dependent]).astype(np.float).tolist())

    from sklearn.preprocessing import normalize
    data_x = normalize(data_x, axis=1)

    from sklearn.preprocessing import scale
    data_x = scale(data_x, axis=1)

    results = reg_m(data_y, data_x)
    coefficients = results.params

    # print_coefficients(predictor_list, coefficients)
    # print(results.summary())

    return coefficients


def compute_suggested(coefficients, inputs):
    values = []

    for i in range(len(inputs)):
        sum = 0

        for j in range(len(inputs)):
            sum += float(coefficients[j]) * float(inputs[j])

        sum += coefficients[-1]
        sum -= coefficients[i]

        values.append(sum / -coefficients[i])

    return values


def eval(coefficients, inputs):
    sum = 0

    for j in range(len(inputs)):
        sum += coefficients[j] * inputs[j]

    sum += coefficients[-1]

    return sum


def get_suggested(dataset_source, x_list, y, inputs):
    coefficients = get_coefficients(dataset_source, x_list, y)
    suggested = compute_suggested(coefficients, inputs)

    if y == "CASUALTIES":
        suggested = suggested[2:]
    elif y == "DAMAGED HOUSES":
        suggested = suggested[3:]
    else:
        suggested = suggested[4:]

    suggestable = []

    for sug in suggested:
        if (sug > 0):
            suggestable.append(sug)
        else:
            suggestable.append(0)

    return suggestable


def load_file(file_name):
    data = []
    with open(file_name, 'rU') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append(row[0:-2])

    return data


def create_file(file_name, data):
    with open(file_name, 'w') as f:
        for row in data:
            f.write(",".join(str(item) for item in row) + '\n')


# usage
dataset_source1 = "DWTARIMA-CAS.csv"
dataset_source2 = "DWTARIMA-DAH.csv"
dataset_source3 = "DWTARIMA-DAP.csv"

x_list1 = ["WIND", "POP", "AI", "PR", "HP", "HMB"]
x_list2 = ["TYPE", "DURATION", "WIND", "AI", "HMB", "HMC"]
x_list3 = ["WIND", "TYPE", "DURATION", "INTENSITY", "AI", "PR"]
y1 = "CASUALTIES"
y2 = "DAMAGED HOUSES"
y3 = "DAMAGED PROPERTIES"

cas = load_file("CAS_suggested.csv")
dah = load_file("DAH_suggested.csv")
dap = load_file("DAP_suggested.csv")

newcas = []
for row in cas:
    suggested = get_suggested(dataset_source1, x_list1, y1, row)
    newcas.append(row + suggested)

create_file("cas_suggest.csv", newcas)

newdah = []
for row in cas:
    suggested = get_suggested(dataset_source2, x_list2, y2, row)
    newdah.append(row + suggested)

create_file("dah_suggest.csv", newdah)

newdap = []
for row in cas:
    suggested = get_suggested(dataset_source3, x_list3, y3, row)
    newdap.append(row + suggested)

create_file("dap_suggest.csv", newdap)
