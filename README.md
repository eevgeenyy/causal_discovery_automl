# causal_discovery_automl

Образ результата проекта: веб сервис, позволяющий автоматизировано проводить поиск алгоритмов и гиперпараметров вычисления наиболее вероятного причинного графа по данным наблюдений (causal discovery) и отчет о результатах его работы.   

## Задача минимум: 
- Реализовать набор алгоритмов, описанных в докладе: 
https://www.youtube.com/watch?v=aMwVHIlNnac&list=WL&index=137&t=1317s
- Создать интерфейс, позволяющий загрузить данные, запустить алгоритм и получить итоговый граф вместе с базовыми статистиками 
- Провести оценку результатов работы алгоритма на нескольких датасетах в сравнении с бенчмарком 

## Задача максимум: 
Добиться улучшения производительности и качества работы решения

## Основные понятия: 

Конфигурация - сочетание causal discovery алгоритма и его гиперпараметров

Марковское ограждение (Markov blanket, MB) - выделяемый для отдельный вершины подграф ориентированного графа, для которого невозможно повысить точность предсказания значения в данной вершине путем добавления дополнительных вершин и ребер исходного графа 

Статистическая неотличимость - ситуация, когда два и более каузальных графа (или марковских ограждения) порождают почти эквивалентные значения наблюдаемых переменных 

Снижение плотности графа - сокращение числа ребер ориентированного каузального графа таким образом, чтобы получить граф статистически неотличимый от исходного 

delta_P - разница между средней предсказательной способностью конфигурации a и другой конфигурации a*

## Пайплайн AutoMl сервиса:
- Поиск наиболее оптимальной конфигурации и соответствующего ей каузального графа
- Снижение плотности графа, полученного на шаге 1
- Вычисление итоговой метрики качества конфигурации 

## Алгоритм поиска наиболее подходящего графа:
  For each configuration c:
  
    For each fold k:
  
      Estimate the causal graph on D_train using c
    
      For each node i:
      
        Fit a predictive model f on D_k_train using MB(i)
        
        Evaluate a predictive performance P_cki on D_k_test
      
      P_avr :=  average P over nodes and folds
  
    Find c* that maximizes P_avr
  
  Return a causal graph g produced by c* 


## Алгоритм снижения плотности графа:

  Calculate a set of more sparse graphs based on g by deleting a single edge
  
  g* := random element from the set
    
  While the set is not empty:
  
    Pool predictions from all folds from g and g* and unite them in vectors p and p*, respectively
    
    Create new vectors from p and p* by swapping elements in these vectors with specified probability 
    
    Compute delta_P and p value for it
    
    if p_value <= threshold:
  
      g := g*
    
      Calculate a set of more sparse graphs based on g by deleting a single edge
  
    else:
    
      del g* from the set 
      
      g* := random element from the set
  
  Return g

## Метрика качества конфигурации:

![image](https://user-images.githubusercontent.com/75931145/202871505-ef754b4a-4188-4368-b625-899d90b87dc1.png)

В качестве дополнительной метрики возможно использование расстояния Хэмминга

## Возможные источники датасетов:

- https://data.world/datasets/causal-discovery
- http://www.causality.inf.ethz.ch/challenge.php?page=datasets#cont
- http://www.causality.inf.ethz.ch/CEdata/
- https://github.com/rguo12/awesome-causality-data
- https://causeme.uv.es/
- Kaggle 

## Прочее

В тех случаях, когда истинный каузальный граф не известен (т.е. для любых реальных (несимулированных) данных, полученных не в результате правильно задизайненных и проведенных экспериментов) предполагается создание симулированного датасета на основе реальных данных по следующему алгоритму:
Применение к датасету одного из стандартных алгоритмов поиска каузального графа и оценки влияния связанных вершин друг на друга 
Генерация нового датасета на основе полученных на первом шаге результатов 

В качестве отдельной подзадачи можно оценить перфоманс пайплайна на данных сгенерированных путем нелинейный операций 




