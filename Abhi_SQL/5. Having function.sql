-- Having VS where 

Select gender, avg(age)
from employee_demographics
group by gender
having avg(age) > 40
;


Select occupation, avg(salary)
from  employee_salary
group by occupation
;


Select occupation, avg(salary)
from  employee_salary
where occupation like '%manager%' 
group by occupation
having avg(salary) > 75000
;
-- where function is used in the row level 
-- having function only works after group_by as the agricate function level
 

