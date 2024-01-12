import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    links = corpus[page]
    num_pages = len(corpus)

    if len(links) == 0:
        # If page has no outgoing links, distribute evenly across all pages
        probability = 1 / num_pages
        for p in corpus:
            distribution[p] = probability
    else:
        # Distribute probability between linked pages and random jump
        for p in corpus:
            distribution[p] = (1 - damping_factor) / num_pages
        for link in links:
            distribution[link] += damping_factor / len(links)

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {page: 0 for page in corpus}
    sample = random.choice(list(corpus.keys()))  # First sample chosen randomly

    for _ in range(n):
        page_rank[sample] += 1
        distribution = transition_model(corpus, sample, damping_factor)
        sample = random.choices(list(distribution.keys()), weights=distribution.values(), k=1)[0]

    # Normalize the PageRank values
    for page in page_rank:
        page_rank[page] /= n

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    page_rank = {page: 1 / num_pages for page in corpus}
    new_rank = page_rank.copy()

    # Repeat until convergence
    while True:
        for page in corpus:
            sum_rank = 0
            for p in corpus:
                if page in corpus[p] or len(corpus[p]) == 0:
                    sum_rank += page_rank[p] / (len(corpus[p]) if len(corpus[p]) > 0 else num_pages)
            new_rank[page] = (1 - damping_factor) / num_pages + damping_factor * sum_rank

        # Check for convergence (threshold e.g., 0.001)
        if all(abs(new_rank[p] - page_rank[p]) < 0.001 for p in corpus):
            break
        page_rank = new_rank.copy()

    return new_rank


if __name__ == "__main__":
    main()
