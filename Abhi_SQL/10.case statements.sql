-- case  statement 

select first_name,
last_name,
age,
case 
	when age <= 30 then 'young'
    when age between 31 and 50 then 'old'
	when age >= 50 then "on death's door"
end
from employee_demographics
;

-- pay increase and bonus 
-- <50000 = 5%
-- >50000 = 7%
-- finance = 10% bonus

 
select first_name,
last_name,
salary,
case
	when salary < 50000 then salary + (salary * 0.05)
    when salary > 50000 then salary + (salary * 0.07)
        
end as New_Salary,

case
	when dept_id = 6 then salary + (salary * 0.10)
end as bonus 



from employee_salary
order by bonus desc
;
