-- Group by 

Select * 
from employee_demographics;


Select gender 
from employee_demographics
group by gender
;

Select first_name 
from employee_demographics
group by gender
;

Select gender,avg(age) , max(age), min(age),count(age)
from employee_demographics
group by gender
;

-- Office Manager	50000
-- Office Manager	60000
Select occupation, salary
from employee_salary
group by occupation, salary
;

-- order by 

Select *
from employee_demographics
order by first_name asc
;

Select *
from employee_demographics
order by first_name desc
;


Select *
from employee_demographics
order by gender, age desc
;

Select *
from employee_demographics
order by age, gender 
;

-- 4 is age
-- 5 is gender 
Select *
from employee_demographics
order by 5,4
;
