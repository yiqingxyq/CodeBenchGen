# Start from python
FROM python:3.8.16

# Install
RUN apt-get update
RUN apt-get install -y zsh tmux wget git autojump unzip vim htop less

# Set the working directory inside the container
WORKDIR /home/user

# Copy files 
COPY . /home/user/CodeBenchGen

# Install the required packages
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /home/user/CodeBenchGen/requirements.txt

# override default image starting point (otherwise it starts from python)
CMD /bin/bash
ENTRYPOINT []