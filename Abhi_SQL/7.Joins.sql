-- joins

select * 
from employee_demographics;

select *
from employee_salary;


select *
from employee_demographics
inner join employee_salary
	on employee_demographics.employee_id = employee_salary.employee_id
;

select dem.employee_id, age, occupation
from employee_demographics as dem
inner join employee_salary as sal
	on dem.employee_id = sal.employee_id
;


-- outer joins

select *
from employee_demographics
left join employee_salary
	on employee_demographics.employee_id = employee_salary.employee_id
;


select *
from employee_demographics
right join employee_salary
	on employee_demographics.employee_id = employee_salary.employee_id
;

-- self join 


select *
from  employee_salary emp1
join  employee_salary emp2
	on emp1.employee_id = emp2.employee_id
;

select *
from  employee_salary emp1
join  employee_salary emp2
	on emp1.employee_id + 1 = emp2.employee_id
;

select *
from  employee_salary emp1
join  employee_salary emp2
	on emp1.employee_id + 1 = emp2.employee_id
;

-- multiple joins


select *
from employee_demographics as dem
inner join employee_salary as sal
	on dem.employee_id = sal.employee_id
inner join parks_departments pd
	on sal.dept_id = pd.department_id

;


