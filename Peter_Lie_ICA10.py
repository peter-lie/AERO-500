# Peter Lie
# AERO 500 / 470

# ICA 10


# 1. Get one item of information from the user
user_name = input("Enter a name: ")

# 2. Store and display back information
filename = f"{user_name}_data.txt"
with open(filename, "w") as f:
    f.write("This file is for " + user_name + "\n")
    f.write("This is a second line for " + user_name + "\n")
    f.write(user_name + " just finished his thesis!")

# 3. Read and display content
with open(filename, "r") as f:
    lines = f.readlines()
    print("\nFile content:")
    for line in lines:
        print(line.strip())

# 4. JSON-based student object
StudentJSON = {
    "emplID": 1,
    "name": "Emma",
    "grades": {
        "english": 82,
        "geometry": 74
    }
}

# Can also store and read back the JSON file with these:
import json
with open("StudentJSONexample1.json", "w") as file: # Change filename
    json.dump(StudentJSON, file)
with open("StudentJSONexample1.json", "r") as file:
    StudentJSONread = json.load(file)


# Define Student class
class Student:
    def __init__(self, emplID, name, grades):
        self.emplID = emplID
        self.name = name
        self.grades = grades

    def __str__(self):
        return f"Student {self.name} (ID: {self.emplID}) has grades: {self.grades}"

# Instantiate student from JSON
student = Student(**StudentJSON)

# Print student info
print("\nStudent object from JSON:")
print(student)

# print(StudentJSONread)


