# Use the official Jupyter base image with Python 3.8
FROM jupyter/base-notebook:python-3.8

# Set the working directory in the container to /home/vatsal/work (default for jupyter/base-notebook)
WORKDIR /home/vatsal/work

# Copy the requirements.txt file into the container
COPY requirements.txt /home/vatsal/work

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project into the container
COPY . /home/vatsal/work

# Make port 8888 available outside this container
EXPOSE 8888

# Start the Jupyter notebook server
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]

