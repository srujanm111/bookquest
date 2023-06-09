# BookQuest: Use AI to Answer Questions from Books!

BookQuest leverages the power of ChatGPT and word embeddings to answer questions based on the text of a book. This AI-powered tool can parse any given book text and answer queries related to it. Ideal for students, researchers, or just curious minds, it brings the future of automated reading and learning to the present.

## Features

- Quick and precise answers from any book text.
- Supports various file formats like .txt, .pdf, .docx, etc.
- Easy to use command-line interface.
- Utilizes the advanced AI model, ChatGPT, for best results.

## Installation

Make sure you have Python 3.7+ installed on your machine. Also, make sure that `pip` (Python package installer) is up-to-date.

Clone the repository and install the required packages.

```bash
git clone https://github.com/srujanm111/bookquest.git
cd bookquest
pip install -r requirements.txt
```

## Usage

Once you have everything installed, you can simply run the main script from the command line.

```bash
python main.py --api-key sk-xxx --file your_book.txt --question "Your question goes here?"
```

Replace 'sk-xxx' with your OpenAI API key, 'your_book.txt' with the path to the file of your book, and type your question in place of 'Your question goes here?'.

You will see an answer generated by the AI for your question in the console.
