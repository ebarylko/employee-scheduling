# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dimod import ConstrainedQuadraticModel, Binary
from dwave.system import LeapHybridCQMSampler
from functools import reduce, partial
from operator import itemgetter


# Set the solver we're going to use
def set_sampler():
    '''Returns a dimod sampler'''

    return LeapHybridCQMSampler()

# Set employees and preferences
def employee_preferences():
    '''Returns a dictionary of employees with their preferences'''

    preferences = { "Anna": [1,2,3,4],
                    "Bill": [3,2,1,4],
                    "Chris": [4,2,3,1],
                    "Diane": [4,1,2,3],
                    "Erica": [1,2,3,4],
                    "Frank": [3,2,1,4],
                    "George": [4,2,3,1],
                    "Harriet": [4,1,2,3]}

    return preferences


def potential_shifts(employees):
    """
    Args:
        employees: a collection of employees, each containing a name and their preferences for all shifts

    Returns: a collection of potential shifts for each employee
    """
    name_to_working_shifts = {'Anna': [0, 1, 2]}

    def generate_shifts(employee):
        name, preferences = employee
        shifts = name_to_working_shifts.get(name, [0, 1, 2, 3])
        return {name: [f"{name}_{shift}" for shift in shifts]}

    def concat_employee_info(employee1, employee2):
        return employee1 | employee2

    return reduce(concat_employee_info, map(generate_shifts, employees.items()))


def add_shift_constraint(shift_preferences, model, employee):
    """
    Args:
        shift_preferences: a collection of key value pairs containing the name of an employee and their preferences
        for all the shifts
        employee: a given employee containing their name and shifts they can work

    Returns: returns a new model containing the constraints associated with the current employee
    """
    print(employee)
    name, shifts = employee
    model.add_discrete(shifts, label=f"{name}")
    model.objective.add_linear_from([*zip(shifts, shift_preferences.get(name))])

    return model


def add_bill_and_frank_constraint(model, shifts):
    """
    Args:
        model:
        shifts: a collection of two series, depicting the shifts Bill and Frank can take on

    Returns: adds a new constraint to the model which forbids Bill and Frank from working together on the
    same shift
    """
    def update_shift_constraint(model, shift):
        frank_shift, bill_shift, idx = shift
        x, y = Binary(frank_shift), Binary(bill_shift)
        model.add_constraint_from_model(x + y, "==", 0, f"shift_{idx}")
        return model

    frank_shifts = shifts[0]
    bill_shifts = shifts[1]
    return reduce(update_shift_constraint,
                  zip(frank_shifts, bill_shifts, range(len(bill_shifts))),
                  model)


def add_constraints(model, shifts, shift_preferences):
    """
    Args:
        model: the model used to model the constraints on the shifts
        shifts: a list of key-value pairs containing the employee name and the shifts the employee could take on
        shift_preferences: the preference each employee has towards the potential shifts

    Returns: a model containing constraints reflecting the preferences of the given employees
    """
    add_constraint = partial(add_shift_constraint, shift_preferences)
    updated_model = reduce(add_constraint, shifts.items(), model)
    return updated_model
    # return add_bill_and_frank_constraint(updated_model,
    #                                      itemgetter("Frank", "Bill")(shifts))



# Create CQM object
def build_cqm():
    '''Builds the CQM for our problem'''

    employees = employee_preferences()
    num_shifts = 4

    # Initialize the CQM object
    cqm = ConstrainedQuadraticModel()

    labels = potential_shifts(employees)

    updated_cqm = add_constraints(cqm, labels, employees)
    return updated_cqm
    # for employee, preference in preferences.items():
    #     # Create labels for binary variables
    #     labels = [f"x_{employee}_{shift}" for shift in range(num_shifts)]
    #
    #     # Add a discrete constraint over employee binaries
    #     cqm.add_discrete(labels, label=f"discrete_{employee}")
    #
    #     # Incrementally add objective terms as list of (label, bias)
    #     cqm.objective.add_linear_from([*zip(labels, preference)])


# Solve the problem
def solve_problem(cqm, sampler):
    '''Runs the provided cqm object on the designated sampler'''

    # Initialize the CQM solver
    sampler = set_sampler()

    # Solve the problem using the CQM solver
    sampleset = sampler.sample_cqm(cqm, label='Training - Employee Scheduling')

    # Filter for feasible samples
    feasible_sampleset = sampleset.filter(lambda x:x.is_feasible)

    print(feasible_sampleset)
    return feasible_sampleset

# Process solution
def process_sampleset(sampleset):
    '''Processes the best solution found for displaying'''
   
    # Get the first solution
    sample = sampleset.first.sample

    shift_schedule=[ [] for _ in range(4)]

    # Interpret according to shifts
    for key, val in sample.items():
         if val == 1.0:
            name = key.split('_')[0]
            shift = int(key.split('_')[1])
            shift_schedule[shift].append(name)

    return shift_schedule

## ------- Main program -------
if __name__ == "__main__":

    # Problem information
    shifts = [1, 2, 3, 4]
    num_shifts = len(shifts)

    cqm = build_cqm()

    sampler = set_sampler()

    sampleset = solve_problem(cqm, sampler)

    shift_schedule = process_sampleset(sampleset)
    #
    for i in range(num_shifts):
        print("Shift:", shifts[i], "\tEmployee(s): ", shift_schedule[i])
    #
