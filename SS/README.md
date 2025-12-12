# Imperial College London - Data Science Project Template

> **⚠️ IMPORTANT: When creating your repository from this template, make sure to select "Private" visibility. Your coursework repository should be private, not public.**

This template provides a standardized structure for data science coursework and assignments at Imperial College London. It is adapted from the [World Bank Project Template](https://github.com/worldbank/template) and designed to help you organize your code, data, analysis, and documentation in a professional and reproducible manner.

## What is This Template For?

This template is designed for:
- **Individual coursework assignments**
- **Group projects**
- **Research projects**

It provides a consistent structure that makes your work easier to organize, share with teammates, and submit for assessment.

## Important Guidelines

Before you start, please review:
- 📋 [Submission Guidelines](docs/submission-guidelines.md) - How to prepare your work for submission
- ✅ [Academic Integrity](docs/academic-integrity.md) - Citation, plagiarism, and proper attribution
- 🆘 [Getting Help](docs/getting-help.md) - Where to find support when you're stuck

## Documentation

The `docs/` folder contains helpful guides:
- **[Getting Started with GitHub](docs/getting-started-github.md)** - GitHub Desktop basics and workflows
- **[Git Workflows](docs/git-workflows.md)** - Best practices for commits, branches, and backups
- **[Anaconda Setup Guide](docs/anaconda-setup.md)** - Setting up Python environments with conda
- **[Jupyter Notebook Best Practices](docs/notebooks-workflows.md)** - Organizing and documenting notebooks
- **[Submission Guidelines](docs/submission-guidelines.md)** - Pre-submission checklist and requirements
- **[Academic Integrity](docs/academic-integrity.md)** - Proper citations and AI tool disclosure
- **[Getting Help](docs/getting-help.md)** - Troubleshooting and support resources

## Learning Resources

To help you get started, check these DataCamp resources:

**Jupyter Notebooks:**
- [Jupyter Notebook Tutorial](https://www.datacamp.com/tutorial/tutorial-jupyter-notebook) - Complete guide to using Jupyter notebooks
- [Jupyter Notebook Cheat Sheet](https://www.datacamp.com/cheat-sheet/jupyter-notebook-cheat-sheet) - Quick reference guide

**GitHub and Version Control:**
- [Introduction to Git](https://www.datacamp.com/courses/introduction-to-git) - Version control fundamentals (DataCamp course)
- [GitHub Concepts](https://www.datacamp.com/courses/introduction-to-github-concepts) - Collaboration and workflows (DataCamp course)
- [GitHub Student Developer Pack](https://education.github.com/pack) - Free access to developer tools for students
- [Introduction to GitHub Copilot](https://education.github.com/experiences/intro_to_copilot) - Learn AI pair programming with Copilot

**Development Tools:**
- [Setting Up VS Code for Python](https://www.datacamp.com/tutorial/setting-up-vscode-python) - Complete guide to VS Code for data science (DataCamp tutorial)
- [VS Code for Data Science](https://code.visualstudio.com/docs/datascience/data-science-tutorial) - Official VS Code data science tutorial
- [GitHub Copilot Agent Mode](https://code.visualstudio.com/docs/copilot/chat/chat-agent-mode) - Use AI agents for autonomous coding tasks


## Template Structure

Inspired by [literate programming](http://literateprogramming.com) and built as [GitHub template repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template), the **template** contains:

- **README.md**
    > README files are important and often neglected. This file should inform anyone about the first steps to use and understand your project. When ready to submit, replace this with your completed project README using the [README-EXAMPLE.md](README-EXAMPLE.md) template.

- **README-EXAMPLE.md**
    > Template for your project README. Fill this in with your project details (title, student/group info, executive summary, contributions, AI usage, etc.) and use it to replace README.md before submission.

- **docs/**
    > Documentation is often never prioritized until last minute. The **template** includes comprehensive guides on best practices, submission guidelines, academic integrity, and more to help you succeed in your coursework.

- **data/**
    > Placeholder folder for data. Data is immutable. By default, the data folder is present but ignored from version control, in order to prevent files of being mistakenly versioned in the code repository.

- **src/**
    > Placeholder folder for source code. Contains example Python files (prefixed with `example-`) that you can reference. Add your own reusable Python functions and modules here that you'll import into your notebooks.

- **notebooks/**
    > Placeholder folder for [Jupyter notebooks](https://jupyter.org). Contains example notebooks (prefixed with `example-`) that you can reference. Add your own notebooks here for data exploration, analysis, and visualization work.

- **requirements.txt**
    > List of Python package dependencies installed via pip. After creating your conda environment, install packages with `pip install -r requirements.txt`. See [Anaconda Setup Guide](docs/anaconda-setup.md) for details.


## Why Are We Using This Template?

The world is changing, and data science has unique qualities and patterns that require a departure from traditional PDF reports. As we will be using Python and Jupyter to develop and communicate our analysis, it is natural that we seek to use it as the primary medium of communicating the results.

**Modern data science communication:** Traditional PDF reports separate code from results, making it difficult to reproduce findings or share computational workflows. With Jupyter notebooks and GitHub, code, analysis, and narrative are integrated in a single, reproducible format.

**Tracking individual contributions:** A common issue that has emerged in the past is that it was difficult to establish individual contributions in group projects. GitHub provides a transparent record of how each member of the team contributed to the activity - including who wrote what code and when - that is very simple for us to verify and ensures fair assessment.

## Usage

### Getting Started

> ✨ **Note:** Please ensure you are logged in on [GitHub](https://github.com) and have permissions to create a repository.

#### 1. Create new repository from template

The **template** is a [GitHub template repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template); in other words, you can generate a new GitHub repository with the same files and folders to use as the starting point for your project.

**Steps:**
1. Give your repository a name
2. **Important:** Choose **Private** for the visibility (recommended for coursework)
3. Click **Create repository from template**

*Voilà!* The repository has been created with the same files and folders of the **template**.

> **See also:** For additional information, see the [GitHub documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)

#### 2. Add collaborators (for group projects)

> **⚠️ Important for Group Projects:** A single repository is required per group. Please agree with your team members on who will set it up (remember that it must be private) and then invite the others as collaborators.

> **⚠️ Individual Contributions:** Every team member is expected to make individual contributions to **code, notebooks, and documentation**. Your instructor will assess individual contributions through Git commit history. Make sure everyone commits regularly under their own GitHub account throughout the project. All team members should contribute to coding work - documentation alone is not sufficient.

If you're working in a team, you'll need to add your teammates and instructor as collaborators to give them access to your private repository.

**Steps:**
1. Go to your repository on GitHub
2. Click on **Settings** (top menu bar)
3. Click on **Collaborators** in the left sidebar (under "Access")
4. Click **Add people**
5. Enter your teammate's GitHub username or email
6. Select the appropriate permission level:
   - **Write**: Can read, clone, and push to the repository (recommended for team members)
   - **Admin**: Full access including settings (use for project leads)
7. Click **Add [username] to this repository**
8. **Repeat for all team members**

**Important - Add Your Instructor:**

You must also add Prof Panagiotis Angeloudis as a collaborator - this is the official form of submission:

1. Click **Add people**
2. Enter username: **`p-ang`**
3. Select permission level: **Read** (or Write if specified by instructor)
4. Click **Add p-ang to this repository**

> **💡 Tip:** Consider adding your instructor early in the project (not just at submission time) if you'd like to request guidance or feedback as you work. This allows them to see your progress and provide help when needed.

Your collaborators will receive email invitations and must accept them to gain access.


#### 3. Set up your environment

Create a conda environment for your project:

```bash
# Create conda environment
conda create -n coursework python=3.11

# Activate it
conda activate coursework

# Install packages
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

> **Note:** If you don't have Anaconda installed, see [Anaconda Setup Guide](docs/anaconda-setup.md) for installation instructions.

#### 4. Prepare your project README

When you're ready to submit, you'll need to replace this template README with your own project documentation.

**Use the provided template:**
1. Open [README-EXAMPLE.md](README-EXAMPLE.md) - this is a template for your project README
2. Fill in all the required information:
   - Your name/group members and GitHub usernames
   - Executive summary of your project
   - How to run your code
   - Data sources and citations
   - Individual contributions (for group projects)
   - Statement on use of AI tools
3. When ready to submit, replace this README.md with your completed version


---

**Congratulations!** You've set up your coursework repository. Now you can start working on your project!

## Working on Your Project

### Example Notebooks

The `notebooks/` folder contains example Jupyter notebooks (prefixed with `example-`) demonstrating how to work with API data (World Bank API example).

**These are examples only** - add your own analysis notebooks to the `notebooks/` folder for your coursework. You can delete or keep the example files as reference.

### Organizing Your Work

A typical workflow:
1. **Explore**: Start with data exploration in notebooks
2. **Clean**: Create data cleaning scripts
3. **Analyze**: Perform your main analysis
4. **Document**: Write clear explanations of your methods and findings


## License

This template is licensed under the **MIT License** and is adapted from the [World Bank Project Template](https://github.com/worldbank/template).

**For your coursework:** Your project inherits the MIT License by default, which is fine for coursework. The license doesn't affect private repositories - you maintain full control. If you want to make your project public later or change the license, you can replace the [LICENSE](LICENSE) file.

See the full [LICENSE](LICENSE) file for complete terms and attributions.
