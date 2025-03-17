"""
Utility functions for VQA metric analysis.
This file contains the missing functions referenced in main.py
"""


def get_answers_from_SS(validation_annotation):
    """
    Extract answers from VizWiz annotations.

    Args:
        validation_annotation: JSON object containing VizWiz annotations

    Returns:
        Dictionary mapping question IDs to list of ground truth answers
    """
    ques_id2gt_ans = {}

    for annotation in validation_annotation:
        answers = [ans['answer'].lower() for ans in annotation['answers']
                   if ans['answer_confidence'] == 'yes']

        # Filter out unanswerable questions
        if answers and not all(
                ans in ['unanswerable', 'unsuitable image', 'unsuitable question', 'not answerable'] for ans in
                answers):
            ques_id2gt_ans[annotation['question_id']] = answers

    return ques_id2gt_ans


def get_gtans_count(annotations):
    """
    Count frequency of ground truth answers.

    Args:
        annotations: List of annotation objects

    Returns:
        Dictionary mapping question IDs to dictionaries of answer counts
    """
    ques_id2ans_count = {}

    for anno in annotations:
        question_id = anno['question_id']
        gt_answers = [ans['answer'].lower() for ans in anno['answers']]

        # Create count dictionary for each answer
        ans_count = {}
        for ans in gt_answers:
            if ans in ans_count:
                ans_count[ans] += 1
            else:
                ans_count[ans] = 1

        ques_id2ans_count[question_id] = ans_count

    return ques_id2ans_count