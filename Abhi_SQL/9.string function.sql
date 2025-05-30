-- String functions

select length('skyfall');

select first_name, length(first_name)
from employee_demographics
order by 2
;

select upper('sky')
;
select lower('sky')
;


select first_name, upper(first_name)
from employee_demographics;

select trim('        s     k        y');
select rtrim('                  s k y ');
select ltrim('s k y                   ');

select first_name,
left(first_name,4),
right(first_name,4),
substring(first_name,3,2),
birth_date,
substring(birth_date,6,2) as birth_month 
from employee_demographics;

select first_name, replace(first_name, 'a','z')
from employee_demographics;

select locate('x', 'alexander');


select first_name, locate('an',first_name)
from employee_demographics
order by 2 desc
;
select first_name,last_name,
concat(first_name, '  ',last_name) as full_name  
from employee_demographics;
