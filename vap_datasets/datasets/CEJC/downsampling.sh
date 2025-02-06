
source_dirpath="/data/group1/z40351r/datasets_turntaking/data/CEJC2021"

for file in ${source_dirpath}/audio_8k/*.wav; do
    
    session_name=$(basename "$file" .wav)    

    input="${file}"
    out_dirpath="${source_dirpath}/audio_16k"
    out_path="${out_dirpath}/${session_name}.wav"
    mkdir -p "${out_dirpath}"

    echo "${input}"
    echo "${out_path}"

    sox -G --buffer 32768 "${input}" -r 16000 -b 16 -S "${out_path}"
    
done