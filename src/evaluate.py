import json
import os

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from config import GOLDEN_SET_PATH, RAGAS_RESULTS_PATH
from src.logger import get_logger

logger = get_logger(__name__)

# Golden test set — 10 questions with reference answers
# Based directly on content from Attention Is All You Need
GOLDEN_TEST_SET = [
    {
        "question": "What is the main contribution of the Transformer architecture?",
        "ground_truth": "The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution."
    },
    {
        "question": "How does scaled dot-product attention work?",
        "ground_truth": "Scaled dot-product attention computes the dot products of queries with all keys, divides by the square root of the key dimension dk, applies softmax to get weights, and multiplies by the values."
    },
    {
        "question": "What is multi-head attention and why is it used?",
        "ground_truth": "Multi-head attention linearly projects queries, keys, and values h times with different learned projections, performs attention in parallel, and concatenates results to allow the model to jointly attend to information from different representation subspaces."
    },
    {
        "question": "Why does the Transformer use positional encoding?",
        "ground_truth": "Since the Transformer contains no recurrence or convolution, positional encodings are added to the input embeddings to inject information about the relative or absolute position of tokens in the sequence."
    },
    {
        "question": "What are the three types of attention used in the Transformer?",
        "ground_truth": "The Transformer uses encoder-decoder attention where queries come from the decoder and keys and values come from the encoder, encoder self-attention where all keys values and queries come from the encoder output, and decoder self-attention with masking to prevent leftward information flow."
    },
    {
        "question": "What is the role of the Feed Forward network in each Transformer layer?",
        "ground_truth": "Each encoder and decoder layer contains a fully connected feed-forward network applied to each position separately and identically consisting of two linear transformations with a ReLU activation in between."
    },
    {
        "question": "How does the Transformer achieve better parallelization than RNNs?",
        "ground_truth": "Unlike RNNs which require sequential computation along the length of the sequence, the Transformer uses self-attention which connects all positions with a constant number of operations enabling full parallelization during training."
    },
    {
        "question": "What optimizer and learning rate schedule was used to train the Transformer?",
        "ground_truth": "The Transformer was trained using the Adam optimizer with beta1 of 0.9, beta2 of 0.98, and epsilon of 10 to the power of negative 9, with a learning rate that increased linearly for warmup steps then decreased proportionally to the inverse square root of the step number."
    },
    {
        "question": "What regularization techniques were used in the Transformer?",
        "ground_truth": "The Transformer applied residual dropout to the output of each sub-layer before adding to the sub-layer input and normalizing, dropout to the sums of embeddings and positional encodings, and label smoothing with a value of 0.1."
    },
    {
        "question": "What BLEU score did the Transformer achieve on WMT 2014 English-to-German translation?",
        "ground_truth": "The Transformer achieved 28.4 BLEU on the WMT 2014 English-to-German translation task outperforming all previously reported models including ensembles by more than 2 BLEU."
    }
]


def save_golden_set() -> str:
    """
    Saves the golden test set to data/golden_test_set.json.

    Returns:
        str: path where golden set was saved
    """
    logger.debug("save_golden_set called")

    try:
        os.makedirs(os.path.dirname(GOLDEN_SET_PATH), exist_ok=True)

        with open(GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
            json.dump(GOLDEN_TEST_SET, f, indent=2, ensure_ascii=False)

        logger.info(f"Golden test set saved — {len(GOLDEN_TEST_SET)} questions to {GOLDEN_SET_PATH}")
        return GOLDEN_SET_PATH

    except Exception as e:
        logger.error(f"Failed to save golden set: {e}", exc_info=True)
        raise


def build_ragas_dataset(
    pipeline_fn,
    bm25,
    chunks: list[dict],
    collection,
    llm
) -> Dataset:
    """
    Runs each golden question through the full RAG pipeline
    and collects answers + contexts for RAGAS evaluation.

    Args:
        pipeline_fn: the run_retrieval_pipeline function
        bm25: BM25Okapi index
        chunks: original chunk list
        collection: ChromaDB collection
        llm: ChatGroq instance

    Returns:
        HuggingFace Dataset ready for RAGAS evaluation
    """
    logger.info(f"build_ragas_dataset called — {len(GOLDEN_TEST_SET)} questions")

    from src.generation import run_generation_pipeline

    questions   = []
    answers     = []
    contexts    = []
    ground_truths = []

    for i, item in enumerate(GOLDEN_TEST_SET):
        question = item["question"]
        logger.info(f"Processing question {i+1}/{len(GOLDEN_TEST_SET)}: '{question}'")

        try:
            # Run retrieval
            retrieved_chunks, is_confident = pipeline_fn(
                question, bm25, chunks, collection, llm
            )

            # Run generation
            result = run_generation_pipeline(question, retrieved_chunks, is_confident)

            questions.append(question)
            answers.append(result["answer"])
            contexts.append([c["text"] for c in retrieved_chunks])
            ground_truths.append(item["ground_truth"])

            logger.debug(f"Question {i+1} processed — fallback={result['is_fallback']}")

        except Exception as e:
            logger.error(f"Failed on question {i+1}: {e}", exc_info=True)
            # Add placeholder so dataset stays aligned
            questions.append(question)
            answers.append("ERROR")
            contexts.append([""])
            ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths
    })

    logger.info(f"RAGAS dataset built — {len(questions)} samples")
    return dataset


def run_ragas_evaluation(
    pipeline_fn,
    bm25,
    chunks: list[dict],
    collection,
    llm
) -> dict:
    """
    Runs full RAGAS evaluation on the golden test set.
    Saves results to data/ragas_results.json.

    Metrics:
    - faithfulness: are answers grounded in retrieved context?
    - answer_relevancy: does the answer address the question?
    - context_precision: are retrieved chunks relevant?
    - context_recall: did we retrieve all necessary information?

    Args:
        pipeline_fn: run_retrieval_pipeline function
        bm25: BM25Okapi index
        chunks: original chunk list
        collection: ChromaDB collection
        llm: ChatGroq instance

    Returns:
        dict of metric scores
    """
    logger.info("=== RAGAS Evaluation Started ===")

    try:
        # Build dataset
        dataset = build_ragas_dataset(
            pipeline_fn, bm25, chunks, collection, llm
        )

        # Save golden set
        save_golden_set()

        # Run RAGAS
        logger.info("Running RAGAS metrics — this may take a few minutes")
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
        )

        # Convert to plain dict
        import numpy as np

        scores = {
            "faithfulness":      round(float(np.mean(results["faithfulness"])), 4),
            "answer_relevancy":  round(float(np.mean(results["answer_relevancy"])), 4),
            "context_precision": round(float(np.mean(results["context_precision"])), 4),
            "context_recall":    round(float(np.mean(results["context_recall"])), 4),
        }

        # Save results
        os.makedirs(os.path.dirname(RAGAS_RESULTS_PATH), exist_ok=True)
        with open(RAGAS_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)

        logger.info(f"RAGAS Results: {scores}")
        logger.info(f"Results saved to {RAGAS_RESULTS_PATH}")
        logger.info("=== RAGAS Evaluation Complete ===")
        return scores

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
        raise