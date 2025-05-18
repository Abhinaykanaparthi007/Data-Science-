-- Trigers and events 

select * 
from employee_demographics;

select * 
from employee_salary;
-- batch level or table level triggers will be activated only trigger onces in microsoft sequal server 

Delimiter $$
create trigger employee_insert
	after insert on employee_salary 
    for each row 
begin
	insert into employee_demographics (employee_id, first_name, last_name)
    values (new.employee_id, new.first_name, new.last_name);
end $$
delimiter ;
    
INSERT INTO employee_salary (employee_id, first_name, last_name, occupation,salary, dept_id)
VALUES (13,'Abhinay','Kanaparthi','Data scientist',1000000,null);

select *
from employee_demographics;

select * 
from employee_salary;


-- Events 

select *
from employee_demographics;


delimiter $$
create event delete_retiree
on schedule every 30 second
do
begin
	delete
    from employee_demographics
    where age >= 60;
end $$    
delimiter ;    
    
 select *
 from employee_demographics;
 
 show variables like 'event%';