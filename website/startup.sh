clear
echo 'clearing cached files...'
rm ./uploads/*
rm ./results/*
echo 'starting server...'
python3 -m flask --app main run

