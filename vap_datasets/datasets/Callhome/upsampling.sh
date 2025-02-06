lang="zho"

source_dirpath="/data/group1/z40351r/datasets_turntaking/data/Callhome_${lang}"

for file in ${source_dirpath}/process/8k_wav/*.wav; do
    
    session_name=$(basename "$file" .wav)    

    input="${file}"
    out_dirpath="${source_dirpath}/process/16k_wav"
    out_path="${out_dirpath}/${session_name}.wav"
    mkdir -p "${out_dirpath}"

    echo "${input}"
    echo "${out_path}"

    sox -G --buffer 32768 "${input}" -r 16000 -b 16 -S "${out_path}"
    
done