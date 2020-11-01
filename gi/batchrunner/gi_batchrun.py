#!.venv/bin/python3

# Automate the submission of a batch of GI runs from parameter files in a directory.
#    Copyright (C) 2020  DavidWh.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import requests
from bs4 import BeautifulSoup
import code
import sched
import time
import os
import json
import argparse


SITE_URL = 'https://supercomputing.swin.edu.au/ast80011'
FILE_DOWNLOAD_BASE_URL = 'https://supercomputing.swin.edu.au'

JOB_TYPE_GALAXY1 = 'Galaxy 1'
JOB_TYPE_GALAXY2 = 'Galaxy 2'
JOB_TYPE_MERGE = 'Galaxy Merge'
JOB_TYPE_INTERACT = 'Galaxy Interaction'
JOB_TYPE_PLOT = 'Plot'

JOB_STATUS_COMPLETED = 'Completed'

HTTP_SUCCESS = 200


def logout(requests_session, standard_headers):
    # Log out the user session.
    logout_request = requests_session.get(os.path.join(SITE_URL, 'accounts/logout/'), headers=standard_headers)
    if logout_request.status_code != HTTP_SUCCESS:
        raise Exception(f'Logout failed with status {logout_request.status_code}.')

    print('Logged out')


def login(requests_session, standard_headers, username, password):
    # Log in as username.
    login_page_request = requests_session.get(os.path.join(SITE_URL, 'accounts/login/'), headers=standard_headers)
    if login_page_request.status_code != HTTP_SUCCESS:
        raise Exception(f'Retrieving the login page failed with status {login_page_request.status_code}.')

    # Get the CSRF token from the login page which must be submitted together with the login request.
    login_page_doc = login_page_request.text
    login_page_doc_soup = BeautifulSoup(login_page_doc, 'html.parser')
    login_csrf_token = login_page_doc_soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']

    login_dict = {
        'csrfmiddlewaretoken': (
            None, login_csrf_token), 'username': (
            None, username), 'password': (
                None, password)}
    login_headers = {'Referer': os.path.join(SITE_URL, 'accounts/login/')}

    login_request = requests_session.post(
        os.path.join(
            SITE_URL,
            'accounts/login/'),
        files=login_dict,
        headers={
            **standard_headers,
            **login_headers})
    if login_request.status_code != HTTP_SUCCESS:
        raise Exception(f'Login attempt as {username} failed with status {login_request.status_code}.')

    print(f'Successfully logged in as: {username}')


def run_job_scheduler(scheduler_fn):
    # Main polling loop for running jobs and checking status.
    TASK_STATUS_POLL_INTERVAL = 10.0

    job_scheduler = sched.scheduler()

    def job_scheduler_loop():
        # Run the job loop function.
        if not scheduler_fn():
            return False

        # Re-schedule the loop to run after next_run_delta seconds
        job_scheduler.enter(TASK_STATUS_POLL_INTERVAL, 1, job_scheduler_loop)

    # Set up the scheduler loop.
    job_scheduler.enter(0, 1, job_scheduler_loop)
    job_scheduler.run()


def get_tasks_status(requests_session, standard_headers):
    # Get the status of all the user's jobs on the simulator.
    tasks_page_request = requests_session.get(os.path.join(SITE_URL, 'gi/view_jobs/'), headers=standard_headers)
    if tasks_page_request.status_code != HTTP_SUCCESS:
        raise Exception(f'Requesting job status page failed with status {tasks_page_request.status_code}.')

    tasks_page_doc = tasks_page_request.text
    tasks_page_doc_soup = BeautifulSoup(tasks_page_doc, 'html.parser')
    table_headers = tasks_page_doc_soup.find_all('th')
    tasks_status = {}
    for table_header in table_headers:
        if table_header.text == 'Job Type':
            # HACK: Find the table body for the selected header by traversing the DOM tree up three levels from the table header ('th'),
            # then searching for 'tbody' elements. There is currently no better way to do this because the table contains no other
            # identifying information (e.g. element id).
            table_body = table_header.parent.parent.parent.find('tbody')
            task_rows = table_body.find_all('tr')
            for task_row in task_rows:
                task_status_elements = task_row.find_all('td')
                job_number = int(task_status_elements[2].text)
                job_status = task_status_elements[3].text
                tasks_status[job_number] = job_status
            break

    return tasks_status


def get_files_list(requests_session, standard_headers, model_number):
    # Get the list of all downloadable output files for a model number.
    data_page_request = requests_session.get(
        os.path.join(
            SITE_URL,
            f'gi/view_data/{model_number}/'),
        headers=standard_headers)
    if data_page_request.status_code != HTTP_SUCCESS:
        raise Exception(f'Requesting data directory page failed with status {data_page_request.status_code}.')

    data_page_doc = data_page_request.text
    data_page_doc_soup = BeautifulSoup(data_page_doc, 'html.parser')
    # HACK: Currently just have to assume that all 'h3' elements are data
    # section headers because there is no other way to identify them.
    output_section_headers = data_page_doc_soup.find_all('h3')
    files_list = {}
    for section_header in output_section_headers:
        # Remove status information from the section name
        status_text_position = section_header.text.find(' - Status:')
        trimmed_section_header = section_header.text[0:status_text_position]

        files_list[trimmed_section_header] = {}
        section_table = section_header.find_next('table')
        file_links = section_table.find_all('a')
        for file_link in file_links:
            files_list[trimmed_section_header][file_link.text] = file_link['href']

    return files_list


def read_json_file(file_name):
    # Read a file from disk and process it as JSON.
    with open(file_name, "rb") as json_file:
        return json.load(json_file)


def get_next_job(job_root_dir):
    # Get the next job folder in sequence.
    with os.scandir(job_root_dir) as root_dir_iterator:
        for root_dir_item in root_dir_iterator:
            if not root_dir_item.is_dir():
                continue

            yield root_dir_item


def post_model(requests_session, standard_headers, post_url, model_json):
    # Submit a task to the simulator.
    page_request = requests_session.get(post_url, headers=standard_headers)
    csrf_token = get_embedded_page_csrf_token(page_request.text)
    submit_request = requests_session.post(
        post_url, headers={**standard_headers, **{'X-CSRFToken': csrf_token}}, json=model_json)
    if submit_request.status_code != HTTP_SUCCESS:
        raise Exception(f'POST request to {post_url} failed with status {submit_request.status_code}.')


def get_embedded_page_csrf_token(page_html):
    # The page for each job type contains a CSRF token which needs to be
    # submitted together with the POST request to start the job.

    # HACK: Just do a basic string search for the token which is actually part of an embedded JavaScript script.
    # Couldn't think of a better way to do this without actually running the JS internally, which would just make this
    # all so very, very much more complicated than it needs to be...
    token_begin_str = 'window.CSRF_TOKEN = "'
    token_start_pos = page_html.index(token_begin_str) + len(token_begin_str)
    token_end_pos = page_html.index('"', token_start_pos)
    csrf_token = page_html[token_start_pos:token_end_pos]
    return csrf_token


def check_rerun_model(model_data, job_type, job_status):
    # Detect if the job params have changed from the previous run.
    # We only want to re-run steps where parameters have changed from the last job (plus any subsequent steps).
    if job_type not in job_status.params_dicts.keys() or model_data != job_status.params_dicts[job_type]:
        job_status.rerun_jobs = True


def run_job(requests_session, standard_headers, job_status, model_number):
    # Run a simulation job through each step (from galaxy setup through to final plots).

    job_status.rerun_jobs = False

    # Galaxy 1
    galaxy_1_params = read_json_file(os.path.join(job_status.job_dir.path, 'galaxy1.json'))
    galaxy_1_params['model_number'] = model_number
    check_rerun_model(galaxy_1_params, JOB_TYPE_GALAXY1, job_status)
    job_status.params_dicts[JOB_TYPE_GALAXY1] = galaxy_1_params
    if job_status.rerun_jobs:
        post_model(requests_session, standard_headers, os.path.join(SITE_URL, 'gi/galactic'), galaxy_1_params)
        print(f'Submitted {JOB_TYPE_GALAXY1} task for {job_status.job_dir.path} as model number {model_number}')
    else:
        print(
            f'Skipped re-running {JOB_TYPE_GALAXY1} task for {job_status.job_dir.path} as model number {model_number}')
    yield JOB_TYPE_GALAXY1

    # Galaxy 2
    galaxy_2_params = read_json_file(os.path.join(job_status.job_dir.path, 'galaxy2.json'))
    galaxy_2_params['model_number'] = model_number
    check_rerun_model(galaxy_2_params, JOB_TYPE_GALAXY2, job_status)
    job_status.params_dicts[JOB_TYPE_GALAXY2] = galaxy_2_params
    if job_status.rerun_jobs:
        post_model(requests_session, standard_headers, os.path.join(SITE_URL, 'gi/galactic'), galaxy_2_params)
        print(f'Submitted {JOB_TYPE_GALAXY2} task for {job_status.job_dir.path} as model number {model_number}')
    else:
        print(
            f'Skipped re-running {JOB_TYPE_GALAXY2} task for {job_status.job_dir.path} as model number {model_number}')
    yield JOB_TYPE_GALAXY2

    # Merge Galaxies
    merge_params = read_json_file(os.path.join(job_status.job_dir.path, 'merge.json'))
    merge_params['model_number'] = model_number
    #merge_params['model_numbers'][0] = str(model_number)
    check_rerun_model(merge_params, JOB_TYPE_MERGE, job_status)
    job_status.params_dicts[JOB_TYPE_MERGE] = merge_params
    if job_status.rerun_jobs:
        post_model(requests_session, standard_headers, os.path.join(SITE_URL, 'gi/galacticsmerge'), merge_params)
        print(f'Submitted {JOB_TYPE_MERGE} task for {job_status.job_dir.path} as model number {model_number}')
    else:
        print(f'Skipped re-running {JOB_TYPE_MERGE} task for {job_status.job_dir.path} as model number {model_number}')
    yield JOB_TYPE_MERGE

    # Interact Galaxies
    interact_params = read_json_file(os.path.join(job_status.job_dir.path, 'interact.json'))
    interact_params['model_number'] = model_number
    check_rerun_model(interact_params, JOB_TYPE_INTERACT, job_status)
    job_status.params_dicts[JOB_TYPE_INTERACT] = interact_params
    if job_status.rerun_jobs:
        post_model(requests_session, standard_headers, os.path.join(SITE_URL, 'gi/interactgalaxies'), interact_params)
        print(f'Submitted {JOB_TYPE_INTERACT} task for {job_status.job_dir.path} as model number {model_number}')
    else:
        print(
            f'Skipped re-running {JOB_TYPE_INTERACT} task for {job_status.job_dir.path} as model number {model_number}')
    yield JOB_TYPE_INTERACT

    # Plot Galaxies
    plot_num = 1
    while True:
        plot_params = None
        try:
            plot_params = read_json_file(os.path.join(job_status.job_dir.path, f'plot{plot_num}.json'))
        except FileNotFoundError:
            # No more plots to do.
            break
        plot_params['model_number'] = model_number
        job_type_num = f'{JOB_TYPE_PLOT}{plot_num}'
        check_rerun_model(plot_params, job_type_num, job_status)
        job_status.params_dicts[job_type_num] = plot_params
        if job_status.rerun_jobs:
            post_model(requests_session, standard_headers, os.path.join(SITE_URL, 'gi/plotgalaxies'), plot_params)
            print(f'Submitted {job_type_num} task for {job_status.job_dir.path} as model number {model_number}')
        else:
            print(
                f'Skipped re-running {job_type_num} task for {job_status.job_dir.path} as model number {model_number}')
        yield job_type_num
        # Increment plot_num and keep running until we don't find a matching json params file.
        plot_num += 1


def download_files(requests_session, standard_headers, files_list, output_dir, selected_sections=None):
    # Download output files.
    for section_name, section_files in files_list.items():
        if (selected_sections is not None) and section_name not in selected_sections:
            # If we've only selected particular sections to download then ignore the others.
            continue

        file_dir = os.path.join(output_dir, section_name)
        # Create a new directory for the file (based on the section name) if it doesn't already exist.
        os.makedirs(file_dir, exist_ok=True)
        for file_name, file_url in section_files.items():
            with requests_session.get(os.path.join(FILE_DOWNLOAD_BASE_URL, file_url.lstrip('/\\')), headers=standard_headers, stream=True) as download_request:
                download_request.raise_for_status()
                with open(os.path.join(file_dir, file_name.replace('/', '_')), "wb") as write_output_file:
                    FILE_DOWNLOAD_CHUNK_SIZE = 10000000  # Download only up to 10 MB at a time
                    for file_chunk in download_request.iter_content(chunk_size=FILE_DOWNLOAD_CHUNK_SIZE):
                        write_output_file.write(file_chunk)

            print(f'Downloaded {file_name}')


def job_scheduler_fn(app_state):
    # Checks job state and schedules running next job step or the next job directory.
    tasks_status = get_tasks_status(app_state.requests_session, app_state.standard_headers)
    num_idle_jobs = 0
    for i in app_state.use_models:
        if ((i in tasks_status.keys()) and (tasks_status[i] == JOB_STATUS_COMPLETED)) or (i not in tasks_status.keys()):
            if i not in app_state.all_jobs_state.keys():
                # Initialise Job State information for a model number we haven't used yet in this run.
                app_state.all_jobs_state[i] = JobState()

            # Reset Job State information for the next operation.
            job_state = app_state.all_jobs_state[i]
            job_state.job_type = None
            if job_state.job_step_iter is not None:
                try:
                    # Run the next step of this job.
                    job_state.job_type = next(job_state.job_step_iter)
                except StopIteration:
                    # This job is complete. Download all associated files.
                    job_state.job_type = None
                    job_state.job_step_iter = None
                    files_list = get_files_list(app_state.requests_session, app_state.standard_headers, i)
                    print(f'Downloading files for job {job_state.job_dir.path} (model number {i})')
                    download_files(
                        app_state.requests_session,
                        app_state.standard_headers,
                        files_list,
                        os.path.join(
                            job_state.job_dir.path,
                            'output'))

            if job_state.job_type is None:
                # Starting a new job.
                try:
                    # Get the next job directory.
                    while True:
                        job_state.job_dir = next(app_state.jobs_iterator)
                        if not os.path.isdir(os.path.join(job_state.job_dir.path, 'output')):
                            # No results exist for this job. Start it.
                            break

                        # Results have already been downloaded for this job. Skip it.
                        print(f'Job {job_state.job_dir.path} already has results. Skipping...')
                        job_state.job_dir = None
                except StopIteration:
                    # No more jobs to do.
                    job_state.job_dir = None
                    num_idle_jobs += 1
                    continue

                print(f'Starting job {job_state.job_dir.path}')
                job_state.job_step_iter = run_job(app_state.requests_session, app_state.standard_headers, job_state, i)
                job_state.job_type = next(job_state.job_step_iter)
        else:
            print(f"Model number {i} is waiting for a job to complete... (current task status is: {tasks_status[i]})")

    if num_idle_jobs >= len(app_state.use_models):
        return False  # All jobs completed. Exit the application.

    return True


class AppState:
    def __init__(self):
        self.standard_headers = {'Host': 'supercomputing.swin.edu.au', 'Origin': 'https://supercomputing.swin.edu.au'}
        self.requests_session = requests.Session()
        self.username = None
        self.password = None
        self.all_jobs_state = {}
        self.jobs_iterator = None
        self.use_models = None


class JobState:
    def __init__(self):
        self.job_type = None
        self.job_dir = None
        self.job_step_iter = None
        self.params_dicts = {}


def parseargs():
    # Process command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--jobs',
        required=True,
        help='Directory containing job subfolders with json parameters files to use. Will also receive results output.')
    parser.add_argument(
        '--models',
        nargs="+",
        type=int,
        default=[
            1,
            2,
            3],
        choices=range(
            1,
            4),
        help='(Optional) Specify which of the available model numbers to use. Default is 1 2 3 (all models).')
    parser.add_argument('--username', required=True, help='Your login username for the AST80011 Simulator webpage.')
    parser.add_argument('--password', required=True, help='Your login password for the AST80011 Simulator webpage.')
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    app_state = AppState()
    app_state.username = args.username
    app_state.password = args.password
    app_state.use_models = args.models
    login(app_state.requests_session, app_state.standard_headers, app_state.username, app_state.password)
    try:
        app_state.jobs_iterator = get_next_job(args.jobs)
        run_job_scheduler(lambda: job_scheduler_fn(app_state))
    finally:
        logout(app_state.requests_session, app_state.standard_headers)


if __name__ == "__main__":
    main()
