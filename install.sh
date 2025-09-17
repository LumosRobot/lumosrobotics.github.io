#!/bin/bash

# install tools
sudo apt-get install ruby-full build-essential 
sudo gem install jekyll bundler
sudo bundle install

# preview website
bundle exec jekyll serve

