for f in `ls configs/t5run`; 
    do python -c "import runt5; runt5.train_t5('$f')"; 
    python -c "import runt5; runt5.test_t5('$f')";
    rm -rf outputs; 
    rm -rf outputs_model;
done