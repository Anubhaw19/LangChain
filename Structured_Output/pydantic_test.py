from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str = 'Anubhaw'
    age: Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt=0, lt=10, default = 5, description = 'A decimal value representing the student score')

new_student = {
                'name':'Anubhaw',
                'age': 25,
                'email': 'anubhaw@gmail.com',
                'cgpa' : 9

              }

student = Student(**new_student) # Pydantic object
print('pydantic object:', student)

student_dict = dict(student)
print('dictonary',student_dict)

student_json = student.model_dump_json()
print('json', student_json)