if command -v arxiv-cli &> /dev/null; then
    echo "arxiv-cli found, proceeding..."
else
    echo "arxiv-cli not found, installing..."
    if command -v cargo &> /dev/null; then 
        echo "Installing with cargo..."
        cargo install arxiv-cli
    else
        echo "Cargo not found, attempting with npm..."
        if command -v npm &> /dev/null; then 
            echo "Installing with npm..."
            npm install @cle-does-things/arxiv-cli
        else
            echo "NPM not found, cannot install arxiv-cli"
            exit 1
        fi
    fi
fi

if [ -d "arxiv-1000-papers/texts" ]; then
    rm -rf arxiv-1000-papers/texts
fi

mkdir -p arxiv-1000-papers/
cd arxiv-1000-papers/

echo "Download 1000 most recent AI papers"

arxiv-cli --category cs.AI --limit 1000 --summary

echo "Done!"