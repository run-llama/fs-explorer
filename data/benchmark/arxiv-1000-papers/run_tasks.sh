for i in {2..5}
do
    echo "Processing ${i}/5"
    run-eval -df question_and_answers_${i}.json --advanced --output results_${i}.json
    echo "Sleeping..."
    sleep 60
done