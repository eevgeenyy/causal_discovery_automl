from sqlalchemy.orm import Session
import models
import schemas

def create_dataset(data, db:Session):
    #dataset: schemas.DatasetBase):
    """
    dataset = models.Datasets(
        dataset_link=dataset.dataset_link,
        rows_number=dataset.rows_number,
        columns_number = dataset.columns_number,
        baseline = dataset.baseline
       # has_categorical = dataset.has_categorical,
       # has_numerical = dataset.has_numerical
    )"""
    db.add(data)
    db.commit()
    db.refresh(data)
    #to_do: сделать возврат и логирование ошибки если запись не происхдит


def create_run(run_instance, db:Session):
    """
    run: schemas.RunBase):
    run_scheme = models.Datasets(
        dataset_id= run.dataset_id,
        total_time = run.timedelta,
        best_result = run.best_result,
        best_algorithm = run.best_algorithm,
        best_params = run.best_params
    )"""
    db.add(run_instance)
    db.commit()
    db.refresh(run_instance)



def get_dataset(db:Session, url:str):
    return db.query(models.Datasets).filter(models.Datasets.dataset_link == url).first()

def get_best_run(db:Session, best_runid):
    return db.query(models.Runs).filter(models.Runs.id == best_runid).first()

def update_best_run(best_runid, dataset_id, db:Session):
    dataset = db.query(models.Datasets).filter(models.Datasets.id == dataset_id).first()
    dataset.best_run = best_runid
    db.commit()


