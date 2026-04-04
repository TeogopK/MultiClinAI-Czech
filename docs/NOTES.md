Състезания за обработка на естествен език в биомедицинската област:

1. [MultiClinAI](https://temu.bsc.es/MultiClinAI/) \- Multilingual Clinical Entity Annotation Projection and Extraction
   1. Откриване на медицински термини (заболявания, симптоми, процедури) в клинични текстове на 7 езика (чешки, английски, испански, италиански, румънски, шведски, нидерландски).
   2. Подзадача 1 \- откриване на всички заболявания, симптоми и процедури в текст \- срок 25.03
   3. Подзадача 2 \- откриване на конкретни термини в преведен текст \- срок 9.04
   4. Може да се участва в една или две подзадачи по желание
2. [DETECH](https://detech2026.dei.unipd.it/) \- откриване на термини в биомедицински текстове свързани с оста черва-мозък
   1. Подзадача 1 \- откриване на термини
   2. Подзадача 2 \- генериране на дефиниции на понятията открити в текста
   3. Срок за регистрация \- 13.03
   4. Срок за предаване на двете подзадачи \- 27.03
   5. Може да се участва в една или две подзадачи по желание

3. # [SNOMED CT Entity Linking Benchmark](https://www.drivendata.org/benchmarks/310/benchmark-snomed-ct/) \- официален leaderboard
   1. Откриване и свързване на термини в клинични текстове на английски език
   2. Няма срок

Предимства:

- Добре дефинирана задача и набор от данни за обучение подходящи за курсов проект.
- Възможност да участвате в международно състезание и да публикувате статия.
- Възможност да се запознаете с модерните подходи в обработката на естествен език при решаването на реални задачи.

---

Repo: https://github.com/gmdn/DETECH2026/

---

- Exploratory data analysis
  - Number of texts, terms per type, and language
  - Frequency distribution for each language and entity type
  - Number of sentences, tokens, sentence length, term length
- Term length issue \- loss adjusted to term length
- Zotero \- share all papers in a new library \- [https://www.zotero.org/groups/6458354/enigma/library](https://www.zotero.org/groups/6458354/enigma/library)
- Evaluation scripts and formatting data \- Sylvia
- Eval scripts by organizers \- [https://github.com/nlp4bia-bsc/MultiClinAIEval](https://github.com/nlp4bia-bsc/MultiClinAIEval)
- Github \- [https://github.com/svassileva/enigma_multiclinai](https://github.com/svassileva/enigma_multiclinai)
- Train/dev split for all languages \- Sylvia
- Monolingual models \- each person to research language-specific models that are applicable for biomedical/language-specific and fine tune them for token classification

—------  
Important note from organizers:  
‘Yes, the **texts have some small variations between labels so, at least for now, it won't be possible to train multilabel systems**. The test set will have the exact same setup. This happens for all languages except for Spanish. This is related to the annotation process, let me explain why:  
1\. The original texts were written and annotated in Spanish. The annotation in the other languages was done via annotation projection: we translated all texts and annotations using Machine Translation and tried to match the translated annotations onto the translated texts.  
2\. The result was then revised by bilingual annotators, who used the annotation tool brat's side-by-side view to be able to see the Spanish version of the documents as reference during the process. Since Machine Translation is not perfect, we also instructed the annotators to provide alternative translations to the clinical concepts they were annotating in case they were not correctly translated or idiomatic.  
3\. Once the annotation was finished, we generated a new version of the texts where the annotations that had alternative translations were replaced with the suggested text. However, since the annotators worked on each label independently, sometimes it was not possible to generate one unique version of the text for all entities. This is mostly because some overlapping annotations from different labels would sometimes have clashing translations (e.g. symptoms and procedures have many overlaps, and maybe the procedure would not have a suggestion but the symptom would). It was decided then to create a version of the text for each label.  
In the end, it was a bit of a trade-off between having translated texts of a better quality and having the same version of the text for all labels. For the task, we settled on the former, but there's a version of the data that does not incorporate the suggested translations and could be used for multilabel settings. We may also release this version of the data after the task, for the moment we have decided not to as to not complicate things.”
