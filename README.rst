|test| |codecov| |docs|

.. |test| image:: https://github.com/Intelligent-Systems-Phystech/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/Intelligent-Systems-Phystech/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/Intelligent-Systems-Phystech/ProjectTemplate/master
    :target: https://app.codecov.io/gh/Intelligent-Systems-Phystech/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/Intelligent-Systems-Phystech/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intelligent-systems-phystech.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Дистилляция моделей на многодоменных выборках
    :Тип научной работы: M1P/НИР
    :Автор: Имя Отчество Фамилия
    :Научный руководитель: степень, Фамилия Имя Отчество
    :Научный консультант(при наличии): степень, Фамилия Имя Отчество

Abstract
========

Исследуется проблема понижения сложности аппроксимирующей модели при переносе на новые данные меньшей мощности. Вводятся понятия учителя, ученика для разных наборов данных. При этом мощность одного набора данных больше мощности другого. Рассматриваются методы, основанные на дистилляции моделей машинного обучения. Вводится предположение, что решение оптимизационной задачи от параметров обеих моделей и доменов повышает качество модели ученика. Проводится вычислительный эксперимент на реальных и синтетических данных.


Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/kbayazitov/distillation/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.com/kbayazitov/distillation/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/kbayazitov/distillation/blob/master/code/main.ipynb>`_.
