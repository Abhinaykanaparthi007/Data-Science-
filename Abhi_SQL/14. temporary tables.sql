-- Temporary Tables

create temporary table temp_tlb
(first_name varchar(50),
last_name varchar(50),
favorite_movie varchar(100)
);

select * 
from temp_tlb;


insert into temp_tlb
values('Abhi', 'Kanaparthi', 'KGF2');

insert into temp_tlb
values('Abhishek', 'Kanaparthi', '');


insert into temp_tlb
values('Sirisha', 'G', 'KGF');


select * 
from employee_salary;

create temporary table salary_over_50k 
select * 
from employee_salary
where salary >= 50000;

select * 
from salary_over_50k;