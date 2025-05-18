-- Data Cleaning 

select * 
from layoffs;

-- steps to follow 
-- 1. remove duplicates 
-- 2. standardize the data 
-- 3. null values or blank values 
-- 4. remove any columns 


-- save a copy 
-- create 
create table layoffs_staging 
like layoffs;
-- check
select * 
from layoffs_staging ;
-- insert data 
insert layoffs_staging
select *
from layoffs;


-- 1. remove duplicates 

-- create a cte to find duplicates 
with duplicate_cte as
(
select *,
row_number() over(
partition by company,location,industry,total_laid_off,percentage_laid_off,`date`,stage,country,funds_raised_millions) as row_num
from layoffs_staging
)


select *
from duplicate_cte
where row_num >1;

-- checking the data by company 
select * 
from layoffs_staging
where company = 'Casper';

-- using Cte we can not delect the duplicates so create another table 

CREATE TABLE `layoffs_staging2` (
  `company` text,
  `location` text,
  `industry` text,
  `total_laid_off` int DEFAULT NULL,
  `percentage_laid_off` text,
  `date` text,
  `stage` text,
  `country` text,
  `funds_raised_millions` int DEFAULT NULL,
  `row_num` INT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

select * 
from layoffs_staging2;

-- insert data as in cte 
insert into layoffs_staging2
select *,
row_number() over(
partition by company,location,industry,total_laid_off,percentage_laid_off,`date`,stage,country,funds_raised_millions) as row_num
from layoffs_staging;

select * 
from layoffs_staging2
where row_num > 1;

-- error 1175 safe update mode 
-- orelse edit -> Preferences -> sql editor -> uncheck safe updates 
 SET SQL_SAFE_UPDATES = 0;
 
delete 
from layoffs_staging2
where row_num > 1;

-- removed duplicates 
select * 
from layoffs_staging2;

-- Standardizing data 

select company,trim(company)
from layoffs_staging2;

-- trim just takes out white space from the word

update layoffs_staging2
set company = trim(company);

select distinct industry
from layoffs_staging2
order by 1;

select * 
from layoffs_staging2
where industry like 'crypto%';

update layoffs_staging2
set industry = 'crypto'
where industry like 'crypto%';

select distinct industry 
from layoffs_staging2;

select  * 
from layoffs_staging2
where country like 'united states%'
order by 1;

-- Trailing means removing '.' from united states'.'
-- syntax Trailing '.' from col name 

select distinct country, trim(Trailing '.' from country) 
from layoffs_staging2
order by 1;

Update layoffs_staging2
set country = trim(Trailing '.' from country) 
where country like 'united states%';

-- change date formate from text to date 

select `date`,
str_to_date(`date`,'%m/%d/%Y' )
from layoffs_staging2;


update layoffs_staging2
set `date` = str_to_date(`date`,'%m/%d/%Y' );

select `date`
from layoffs_staging2;

-- only do Alter table on copy tabble not on original, to change formate
 
Alter table layoffs_staging2
modify column`date` date ;

select * 
from layoffs_staging2
where total_laid_off is null 
and percentage_laid_off is null ;

select * 
from layoffs_staging2
where industry is null
or industry = '';

select *  
from layoffs_staging2
where company = 'Airbnb';

-- step 3 blanks and null values 
-- filling missing industry with other availabe data by joining 

select *
from layoffs_staging2 t1
join layoffs_staging2 t2
	on t1.company = t2.company
where (t1.industry is null or t1.industry = '') 
and t2.industry is not null ;


select t1.industry,t2.industry
from layoffs_staging2 t1
join layoffs_staging2 t2
	on t1.company = t2.company
where (t1.industry is null or t1.industry = '') 
and t2.industry is not null ;

-- set all the blanks into null 

update layoffs_staging2
set industry = null
where industry = '';


update layoffs_staging2 t1
join layoffs_staging2 t2
	on t1.company = t2.company
set t1.industry = t2.industry 
where (t1.industry is null ) 
and t2.industry is not null ;

select *  
from layoffs_staging2
where company like 'Bally%';

-- delete unknown data 

select *
from layoffs_staging2
where total_laid_off is null
and percentage_laid_off is null ;

delete 
from layoffs_staging2
where total_laid_off is null
and percentage_laid_off is null ;


-- Step 4 remove any coloumns 
-- drop coloumn row_num we should use alter 

alter table layoffs_staging2
drop column row_num;

select * 
from layoffs_staging2;



