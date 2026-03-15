import os
import requests
from bs4 import BeautifulSoup


def download_flac_files(house, week, year, hours=1, download_dir="~/project"):
    """
    Downloads the first `hours` FLAC files for a given house, week, and year.
    Each file is ~1 hour long.
    Skips files that already exist in the download directory.
    Original filenames from the server are preserved.
    """
    import os, requests
    from bs4 import BeautifulSoup

    # Create the local download directory
    download_dir = os.path.expanduser(f"{download_dir}/house_{house}/flac_files/{year}/wk{week}")
    os.makedirs(download_dir, exist_ok=True)

    # Construct base URL for the server folder
    base_url = (
        f"https://dap.ceda.ac.uk/edc/efficiency/residential/"
        f"EnergyConsumption/Domestic/UK-DALE-2015/UK-DALE-16kHz/"
        f"house_{house}/{year}/wk{week}/"
    )
    
    # Fetch the directory page
    try:
        response = requests.get(base_url)
        response.raise_for_status()
    except requests.HTTPError:
        print(f"ERROR: URL not found: {base_url}")
        return

    # Parse FLAC links
    soup = BeautifulSoup(response.text, "html.parser")
    files = [
        link.get("href")
        for link in soup.find_all("a")
        if link.get("href") and link.get("href").endswith(".flac")
    ]

    if not files:
        print(f"No FLAC files found for house {house}, week {week}, year {year}.")
        return

    # Sort by timestamp (ascending)
    files.sort()

    # Download requested number of files
    for file_name in files[:hours]:
        # Remove "vi-" prefix but keep the rest of the filename
        clean_name = file_name
        if file_name.startswith("vi-"):
            clean_name = file_name[3:]
        # Remove the last _xxx part before .flac
            base = clean_name.split("_")[0]  # drop last part
            clean_name = f"{base}.flac"

        output_path = os.path.join(download_dir, clean_name)

        if os.path.exists(output_path):
            print(f"File already exists, skipping: {output_path}")
            continue

        url = base_url + file_name
        print(f"Downloading {file_name} → {output_path} ...")

        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
        except requests.HTTPError:
            print(f"Failed to download {file_name}")
            continue

        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Download complete: {output_path}")





def download_dat_files(house, channels, download_dir="~/project"):
    """
    Downloads mains.dat + appliance .dat files for a given house.
    Skips files that already exist in the download directory.

    Args:
        house (int): House number
        channels (list[int]): List of appliance channels
        download_dir (str): Directory to save the downloaded files
    """
    download_dir = os.path.expanduser(f"{download_dir}/house_{house}/dat_files/")
    os.makedirs(download_dir, exist_ok=True)

    base_url = f"https://dap.ceda.ac.uk/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2015/UK-DALE-disaggregated/house_{house}/"


    # Download appliance channels
    for channel in channels:
        output_path = os.path.join(download_dir, f"house{house}_channel{channel}.dat")
        if os.path.exists(output_path):
            print(f"File already exists, skipping: {output_path}")
            continue

        url = f"{base_url}channel_{channel}.dat"
        print(f"Downloading channel {channel} → {output_path} ...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Download complete for house {house}, channel {channel}!")
        except requests.HTTPError:
            print(f"ERROR: File not found for house {house}, channel {channel}: {url}")
