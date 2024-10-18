# [ScreenRPA](https://canela.lsi.us.es/rim/api/v1/docs)


## Table of Contents

* [Demo](#demo)
* [Quick Start](#quick-start)
* [Documentation](#documentation)
* [File Structure](#file-structure)
* [Browser Support](#browser-support)
* [Resources](#resources)
* [Reporting Issues](#reporting-issues)
* [Technical Support or Questions](#technical-support-or-questions)
* [Licensing](#licensing)
* [Useful Links](#useful-links)

<br />

## Demo

> To authenticate use the default credentials ***test / ApS12_ZZs8*** or create a new user on the **registration page**.

- **Screen RPA** [Login Page](https://canela.lsi.us.es/screenrpa)
- **[Screen RPA Public Samples](https://canela.lsi.us.es/screenrpa/results)** - sample experiments that show app funtionality 

<br />

## Quick start

> UNZIP the sources or clone the private repository. After getting the code, open a terminal and navigate to the working directory, with product source code.

The following software is needed to run this platform:
- C++ Dev Tools from Visual Studio: ![visual_studio_c++_features](apps\static\assets\img\image.png) 

- If you are on windows, you must install MS Word. On linux, libreoffice writter is needed. To install only the bare necessary you can run `apt install libreoffice-core-nogui libreoffice-writer-nogui --no-install-recommends --no-install-suggests` (Not needed if docker is used)

- Graphviz: 
    - Windows:
    1. Download and install Graphviz 2.46.0 for Windows 10 (64-bit):         [stable_windows_10_cmake_Release_x64_graphviz-install-2.46.0-win64.exe](https://gitlab.com/graphviz/graphviz/-/package_files/6164164/download).
    2. Install PyGraphviz via:   

        ```ps
        python -m pip install --config-settings="--global-option=build_ext" --config-settings="--global-option=-IC:\Program Files\Graphviz\include" --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" pygraphviz
        ```
    - Linux: run `sudo apt-get install graphviz graphviz-dev` (Not needed if docker is used)


```bash
$ # Get the code
$ git clone https://github.com/RPA-US/screenrpa.git
$ cd screenrpa
$
$ # Virtualenv modules installation (Unix based systems)
$ virtualenv env
$ source env/bin/activate
$
$ # Virtualenv modules installation (Windows based systems)
$ # virtualenv env
$ # .\env\Scripts\activate
$
$ # Install modules - SQLite Storage
$ pip3 install -r requirements.txt
$
$ # Create tables
$ python manage.py makemigrations
$ python manage.py makemigrations apps_analyzer apps_behaviourmonitoring apps_decisiondiscovery apps_featureextraction apps_processdiscovery apps_reporting apps_notification
$ python manage.py migrate
$
$ # Populate UI detection models
$ python manage.py loaddata configuration/models_populate.json
$
$ # Start the application (development mode)
$ python manage.py runserver # default port 8000
$
$ # Start the app - custom port
$ # python manage.py runserver 0.0.0.0:<your_port>
$
$ # Access the web app in browser: http://127.0.0.1:8000/
```

> Note: To use the app, please access the registration page and create a new user. After authentication, the app will unlock the private pages.

<br />

## Documentation
The documentation of this project can be found in the [Wiki](https://github.com/RPA-US/rim/wiki/Deployment-instruction-of-RIM-for-a-local-development-environment) associated to this Github Repository.

<br />

## Code-base structure

The project is coded using a simple and intuitive structure presented bellow:

```bash
< PROJECT ROOT >
   |
   |-- core/                               # Implements app configuration
   |    |-- settings.py                    # Defines Global Settings
   |    |-- wsgi.py                        # Start the app in production
   |    |-- urls.py                        # Define URLs served by all apps/nodes
   |
   |-- apps/
   |    |
   |    |-- home/                          # A simple app that serve HTML files
   |    |    |-- views.py                  # Serve HTML pages for authenticated users
   |    |    |-- urls.py                   # Define some super simple routes  
   |    |
   |    |-- authentication/                # Handles auth routes (login and register)
   |    |    |-- urls.py                   # Define authentication routes  
   |    |    |-- views.py                  # Handles login and registration  
   |    |    |-- forms.py                  # Define auth forms (login and register) 
   |    |
   |    |-- static/
   |    |    |-- <css, JS, images>         # CSS files, Javascripts files
   |    |
   |    |-- templates/                     # Templates used to render pages
   |         |-- includes/                 # HTML chunks and components
   |         |    |-- navigation.html      # Top menu component
   |         |    |-- sidebar.html         # Sidebar component
   |         |    |-- footer.html          # App Footer
   |         |    |-- scripts.html         # Scripts common to all pages
   |         |
   |         |-- layouts/                   # Master pages
   |         |    |-- base-fullscreen.html  # Used by Authentication pages
   |         |    |-- base.html             # Used by common pages
   |         |
   |         |-- accounts/                  # Authentication pages
   |         |    |-- login.html            # Login page
   |         |    |-- register.html         # Register page
   |         |
   |         |-- home/                      # UI Kit Pages
   |              |-- index.html            # Index page
   |              |-- 404-page.html         # 404 page
   |              |-- *.html                # All other pages
   |
   |-- resources/
   |         |--models/                     # Models used in apps
   |         |--inputs/                     # Inputs, such as images, used for testing
   |
   |-- requirements.txt                     # Development modules
   |
   |-- .env                                 # Inject Configuration via Environment
   |-- manage.py                            # Start the app - Django default start script
   |
   |-- ************************************************************************
```

<br />

> The bootstrap flow

- Django bootstrapper `manage.py` uses `core/settings.py` as the main configuration file
- `core/settings.py` loads the app magic from `.env` file
- Redirect the guest users to Login page
- Unlock the pages served by *app* node for authenticated users

<br />

> Paths

In this code, the following paths are used:
- root_path: media/unzipped/case_study_XXXXXXXXX/executions/exec_XX/
- scenario_path: media/unzipped/case_study_XXXXXXXXX/executions/exec_XX/scenario1/
- log_path: media/unzipped/case_study_XXXXXXXXX/executions/exec_XX/scenario1/log.csv
- scenario_results_path:  media/unzipped/case_study_XXXXXXXXX/executions/exec_XX/scenario1_results/

<br />




## Recompile CSS

To recompile SCSS files, follow this setup:

<br />

**Step #1** - Install tools

- [NodeJS](https://nodejs.org/en/) 12.x or higher
To install node in this environment
    - `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash` (install NVM (Node Version Manager))
Close and reopen the terminal
    - `nvm --version` (to check nvm is installed ok)
    - `nvm install 12` (Version 12 is full compatible)
    - `node -v` (to check node is installed ok)
- [Gulp](https://gulpjs.com/) - globally 
    - `npm install -g gulp-cli`
- [Yarn](https://yarnpkg.com/) (optional) 

<br />

**Step #2** - Change the working directory to `assets` folder

```bash
$ cd apps/static/assets
```

<br />

**Step #3** - Install modules (this will create a classic `node_modules` directory)

```bash
$ npm install
// OR
$ yarn
```

<br />

**Step #4** - Edit & Recompile SCSS files 

```bash
$ gulp scss
```

Remember to refresh (CTRL + F5) the app to see applied styles.

The generated file is saved in `static/assets/css` directory.

<br /> 

## Deployment

The app is provided with a basic configuration to be executed in [Docker](https://www.docker.com/), [Gunicorn](https://gunicorn.org/), and [Waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/).

### [Docker](https://www.docker.com/) execution
---

The application can be easily executed in a docker container. The steps:

> Get the code

```bash
$ git clone https://github.com/RPA-US/rim.git
$ cd rim
```

> Create a .env file

Copy the .env.sample in the docker folder and replace the values for those you desire.

> Build the container for the app in Docker

There are four images for this application, two for development  and production for systems with nvidia GPUs and two for development and production for those without an nvidia GPU.

```bash
$ sudo docker-compose -f ./docker/<docker-compose-file> up
```

> Start the application

Follow the instructions to run the applications from the "Create tables" section onwards.

If you want to use celery run the following commands:

```bash
$ redis-server
$ python -m celery -A core worker --concurrency=1
```

Visit `http://localhost:85` in your browser. The app should be up & running.

<br />

## Eye Tracking Use and Support

To record a GazeLog with the eye tracking software integrated into ScreenRPA (WebGazer.js <https://github.com/brownhci/WebGazer>), follow these steps:

1. **Start the ScreenRPA Web Application**
   - Launch the ScreenRPA web application.

2. **Access WebGazer**
   - Click on "WebGazer.js" to open the WebGazer.js suite.

3. **Prepare Your Case Study Scenario**
   - Arrange all the applications you will interact with in windowed mode.
   - Ensure the ScreenRPA tab in your browser is open and in the background.

4. **Start Recording**
   - Begin recording the UI Log using either StepRecorders or Screen Action Logger.
   - Then, click on "Start" in the WebGazer suite to record the Gaze Log.

5. **Perform Your Activity**
   - Stay calm and perform your activity naturally. ScreenRPA will handle the rest, recording both your UI log and Gaze Log!

By following these steps, you will effectively utilize the eye tracking capabilities of ScreenRPA to capture comprehensive logs of your interactions.

<br />

## Browser Support

At present, we officially aim to support the last two versions of the following browsers:

<img src="https://s3.amazonaws.com/creativetim_bucket/github/browser/chrome.png" width="64" height="64"> <img src="https://s3.amazonaws.com/creativetim_bucket/github/browser/firefox.png" width="64" height="64"> <img src="https://s3.amazonaws.com/creativetim_bucket/github/browser/edge.png" width="64" height="64"> <img src="https://s3.amazonaws.com/creativetim_bucket/github/browser/safari.png" width="64" height="64"> <img src="https://s3.amazonaws.com/creativetim_bucket/github/browser/opera.png" width="64" height="64">

<br />

## Resources

- Demo: <https://canela.lsi.us.es/rim>
- Documentation: <https://canela.lsi.us.es/rim/api/v1/redoc>
- License Agreement: <https://creativecommons.org/licenses/by-nc/4.0/>
- Support: <https://es3.us.es>
- Issues: [Github Issues Page](https://github.com/RPA-US/rim/issues)

<br />

## Reporting Issues

We use GitHub Issues as the official bug tracker for the **Screen RPA**. Here are some advices for our users that want to report an issue:

1. Make sure that you are using the latest version of the **Screen RPA**.
2. Providing us reproducible steps for the issue will shorten the time it takes for it to be fixed.
3. Some issues may be browser-specific, so specifying in what browser you encountered the issue might help.

<br />

## Technical Support or Questions

If you have questions or need help integrating the product please [contact us](mailto:amrojas@us.es) instead of opening an issue.

<br />

## Licensing

- Copyright CENIT-ES3
- Licensed under [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/)

<br />

## Useful Links

- [Tutorials](https://www.youtube.com/channel/UCq9H-Yj-C9nFifDsH5JwoeA)

<br />

## Social Media

- Twitter: <https://twitter.com/rpa_us>

<br />

---
This platform templates are based on the [ones](https://www.creative-tim.com/product/argon-dashboard-django) provided by [Creative Tim](https://www.creative-tim.com/) and [AppSeed](https://appseed.us)
