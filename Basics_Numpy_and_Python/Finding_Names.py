people = ["Noara Razzak", "Sadman Alvi", "Fabio Hays", "Haroon Goodman",
          "Sarah Sweeney", "Gabrielle Hurst", "Santino Pacheco", "Harvie Goodwin",
          "Timothy Carr", "Qasim Montes", "Lester Ingram", "Blossom Powers",
          "Rohmotullah"]

search_term = "Rohmot"


def person_exists(people, search_term):
    for i in range(len(people)):
        each_name = people[i].split()
        for name in each_name:
            if search_term == name:
                return people[i]
    return None


def person_exists_v2(people, search_term):
    for person in people:
        if search_term in person:
            return person
    return None


matched_name = person_exists_v2(people, search_term)


def name_of_person(matched_name):
    if matched_name:
        print("Person exists. " + "Name is " + matched_name)
    else:
        print("No Match")


def main():
    person_exists(people, search_term)
    person_exists_v2(people, search_term)
    name_of_person(matched_name)