# FCM Preprocessor

This directory contains the Java code required for preprocessing a text corpus so that the form-context model can be trained on it. Note that this code does not perform any form of preprocessing other than distributing the corpus over multiple files in a specific format. You must take care of other preprocessing steps such as lowercasing and tokenization **before** executing this preprocessor.
You can create an executable jar file from the sources using `mvn clean install`. A list of all arguments is available via the `--help` flag.