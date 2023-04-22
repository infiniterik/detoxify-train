export WANDB_API_KEY=df42ca7f17cf94f3ddfa7caa79a9587abbbf1043
for f in `ls configs/t5run`; 
    do 
    export T5RUN=configs/t5run/$f
    #python runt5.py
    python -c "import runt5; runt5.test_t5('configs/t5run/$f', 500)";
    rm -rf outputs; 
    rm -rf outputs_model;
done