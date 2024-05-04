import base64
import io
import os
import pickle
from tempfile import NamedTemporaryFile
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
import datetime
import docx
from docx import Document
from docx.shared import Inches
from django.core.exceptions import ValidationError
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from django.views.generic import ListView, DetailView, CreateView
import pandas as pd
from sklearn.tree import export_graphviz
from .models import PDD
from apps.analyzer.models import CaseStudy
from django.utils.translation import gettext_lazy as _
from apps.analyzer.models import CaseStudy, Execution # add this
from .forms import ReportingForm
from django.urls import reverse

import os
import zipfile
from django.http import HttpResponse
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _
from django.conf import settings
from django.shortcuts import get_object_or_404

from docx import Document
import os
import pydotplus

# Create your views here.

def pdd_define_style(document):
    # =======================================================================================================
    # add a custom style for headings
    heading_style = document.styles.add_style('Heading', WD_STYLE_TYPE.PARAGRAPH)
    heading_style.base_style = document.styles['Heading 1']
    heading_style.font.name = 'Calibri Light'
    heading_style.font.size = docx.shared.Pt(18)
    heading_style.font.color.rgb = docx.shared.RGBColor(0x2d, 0x89, 0xff)

    # add a custom style for the table of contents heading
    toc_heading_style = document.styles.add_style('indice', WD_STYLE_TYPE.PARAGRAPH)
    toc_heading_style.base_style = document.styles['Heading']
    toc_heading_style.font.color.rgb = docx.shared.RGBColor(0x2d, 0x89, 0xff)
    toc_heading_style.font.bold = True
    toc_heading_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # add a custom style for section headings
    section_heading_style = document.styles.add_style('Section Heading', WD_STYLE_TYPE.PARAGRAPH)
    section_heading_style.base_style = document.styles['Heading 2']
    section_heading_style.font.name = 'Calibri Light'
    section_heading_style.font.size = docx.shared.Pt(14)
    section_heading_style.font.color.rgb = docx.shared.RGBColor(0x2d, 0x89, 0xff)
    section_heading_style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    section_heading_style.paragraph_format.line_spacing = 1.5

    # add a custom style for body text
    body_style = document.styles.add_style('normal', WD_STYLE_TYPE.PARAGRAPH)
    body_style.base_style = document.styles['Normal']
    body_style.font.name = 'Calibri'
    body_style.font.size = docx.shared.Pt(12)
    # =======================================================================================================
    
    # Create a custom document style
    doc_styles = document.styles
    custom_style = doc_styles.add_style('CustomStyle', 1)
    custom_style.font.name = 'Segoe UI'
    custom_style.font.size = 12
    
    return document, custom_style

def pdd_define_properties(document):
    # add document properties
    document.core_properties.author = "Antonio Martinez Rojas"
    document.core_properties.title = _("Process Definition Document")
    document.core_properties.comments = _("This is a first version of your RPA Process Definition Document.")
    document.core_properties.category = _("PDDs")
    document.core_properties.created = datetime.datetime.now()
    
    return document
    
def pdd_add_cover(document, custom_style):
    
    # Set the page orientation to portrait
    section = document.sections[0]
    section.orientation = 1
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    
    # Add a cover page
    document.add_picture('cover_image.png', width=Inches(6))
    document.add_paragraph(_('Process Definition Document'), custom_style).alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph(_('Version 1.0'), custom_style).alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_page_break()
    
    return document
    
def pdd_table_of_contents(document):
    # add a table of contents
    paragraph = document.add_paragraph()
    run = paragraph.add_run(_('Table of Contents'))
    paragraph.style = 'indice'
    # paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Inches(0.2)
    paragraph.paragraph_format.space_before = Inches(0.2)
    # Code for making Table of Contents
    paragraph = document.add_paragraph()
    run = paragraph.add_run()

    fldChar = OxmlElement('w:fldChar')  # creates a new element
    fldChar.set(qn('w:fldCharType'), 'begin')  # sets attribute on element

    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')  # sets attribute on element
    instrText.text = 'TOC \\o "1-3" \\h \\z \\u'   # change 1-3 depending on heading levels you need

    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')

    fldChar3 = OxmlElement('w:t')
    fldChar3.text = _("Right-click to update field.")

    fldChar2.append(fldChar3)

    fldChar4 = OxmlElement('w:fldChar')
    fldChar4.set(qn('w:fldCharType'), 'end')

    r_element = run._r
    r_element.append(fldChar)
    r_element.append(instrText)
    r_element.append(fldChar2)
    r_element.append(fldChar4)

    p_element = paragraph._p
    document.add_page_break()
    
    return document

def ui_screen_trace_back_reporting(case_study_id):
    # create a new document
    document = Document()
    
    document = pdd_define_properties(document)
    document, custom_style = pdd_define_style(document)
    document = pdd_add_cover(document, custom_style)
    document = pdd_table_of_contents(document)
    
    # SECTIONS

    # add section 1: introduction
    document.add_heading(_('1. Introduction'), level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('1. Introduction')

    # add body text for section 1
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run(_('This document outlines the process definition for...'))
    document.add_page_break()

    # add section 2: process overview
    document.add_heading('2. Process Overview', level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('2. Process Overview')

    # add body text for section 2
    body_text = document.add_paragraph()
    body_text.style = 'normal'
    body_text.add_run(_('This process consists of the following steps:'))
    # add a numbered list for the steps in section 2
    numbered_list = document.add_paragraph()
    numbered_list.style = 'List Number'
    numbered_list.add_run(_('Step 1: ...'))
    numbered_list.add_run(_('\nStep 2: ...'))
    numbered_list.add_run(_('\nStep 3: ...'))
    
    # add section 3: roles and responsibilities
    document.add_page_break()
    document.add_heading(_(), level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('3. Roles and Responsibilities')

    # add body text for section 3
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run(_('The following roles and responsibilities are involved in this process:'))

    # add a table for the roles and responsibilities in section 3
    table = document.add_table(rows=4, cols=2)
    table.style = 'Table Grid'
    table.autofit = False

    # set the width of the columns in the table
    column_widths = (Inches(1.5), Inches(4))
    for i, width in enumerate(column_widths):
        table.columns[i].width = width

    # add the header row for the table
    heading_cells = table.rows[0].cells
    heading_cells[0].text = _('Role')
    heading_cells[1].text = _('Responsibilities')

    # add the content for the remaining rows in the table
    row_cells = table.rows[1].cells
    row_cells[0].text = _('Manager')
    row_cells[1].text = _('Approves final output of the process')
    row_cells = table.rows[2].cells
    row_cells[0].text = _('Analyst')
    row_cells[1].text = _('Conducts analysis for the process')
    row_cells = table.rows[3].cells
    row_cells[0].text = _('Developer')
    row_cells[1].text = _('Implements changes to the process')

    # add section 4: process flow
    document.add_page_break()
    document.add_heading(_('4. Process Flow'), level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('4. Process Flow')

    # add body text for section 4
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run(_('The process flow for this process is as follows:'))
    # add an image for the process flow in section 4
    document.add_picture('process_flow.png', width=Inches(6))
    # add section 5: process metrics
    document.add_page_break()
    document.add_heading(_('5. Process Metrics'), level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('5. Process Metrics')

    # add body text for section 5
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run(_('The following metrics are tracked for this process:'))
    # add a bulleted list for the metrics in section 5
    bulleted_list = document.add_paragraph()
    bulleted_list.style = 'List Bullet'
    # add the metrics to the bulleted list
    bulleted_list.add_run(_('Metric 1: ...'))
    bulleted_list.add_run(_('\nMetric 2: ...'))
    bulleted_list.add_run(_('\nMetric 3: ...'))
    
    # add section 6: process improvement
    document.add_page_break()
    document.add_heading(_('6. Process Improvement'), level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('6. Process Improvement')

    # add body text for section 6
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run(_('The following improvement opportunities have been identified for this process:'))
    # add a numbered list for the improvement opportunities in section 6

    # Adding list of style name 'List Number'
    document.add_heading('Style: List Number', 3)
    # Adding points to the list named 'List Number'
    document.add_paragraph(_('Opportunity 1: ...'), style='List Number')
    document.add_paragraph(_('Opportunity 2: ...'), style='List Number')
    document.add_paragraph(_('Opportunity 3: ...'), style='List Number')
    
    # Adding list of style name 'List Number 2'
    document.add_heading('Style: List Number 2', 3)
    # Adding points to the list named 'List Number 2'
    document.add_paragraph(_('Opportunity 1: ...'), style='List Number 2')
    document.add_paragraph(_('Opportunity 2: ...'), style='List Number 2')
    document.add_paragraph(_('Opportunity 3: ...'), style='List Number 2')
    
    # Adding list of style name 'List Number 3'
    document.add_heading('Style: List Number 3', 3)
    # Adding points to the list named 'List Number 3'
    document.add_paragraph(_('Opportunity 1: ...'), style='List Number 3')
    document.add_paragraph(_('Opportunity 2: ...'), style='List Number 3')
    document.add_paragraph(_('Opportunity 3: ...'), style='List Number 3')    


    # add section 7: glossary
    document.add_page_break()
    document.add_heading(_(), level=1).style = 'Section Heading'
    # section_heading = document.add_paragraph()
    # section_heading.style = 'Section Heading'
    # section_heading.add_run('7. Glossary')
    
    # add body text for section 7
    body_text = document.add_paragraph()
    body_text.style = 'Body Text'
    body_text.add_run(_('The following terms are used throughout this document:'))
    
    # add a table for the glossary in section 7
    table = document.add_table(rows=3, cols=2)
    table.style = 'Table Grid'
    table.autofit = False
    # set the width of the columns in the table

    column_widths = (Inches(1.5), Inches(4))

    for i, width in enumerate(column_widths):
            table.columns[i].width = width
            
    # add the header row for the table
    heading_cells = table.rows[0].cells
    heading_cells[0].text = _('Term')
    heading_cells[1].text = _('Definition')
    
    # add the content for the remaining rows in the table
    row_cells = table.rows[1].cells
    row_cells[0].text = _('Process')
    row_cells[1].text = _('A set of activities that transform inputs into outputs')
    row_cells = table.rows[2].cells
    row_cells[0].text = _('Metric')
    row_cells[1].text = _('A quantifiable measure used to track the performance of a process')

    # save the document
    document.save('Process Definition Document.docx')
    
    return document
    
    
# TODO:
class ReportGenerateView(DetailView):
    def get(self, request, *args, **kwargs):
        print("se ejecuta")
        cs = get_object_or_404(PDD, id=kwargs["case_study_id"], active=True)
        
        document = ui_screen_trace_back_reporting(cs.id)
        
        # check if the file exists
        if not os.path.exists(document.file.path):
            return HttpResponse("File does not exist", status=404)

        # set the file content type and headers
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{document.name}"'
        
        # read the file and write it to the response
        with open(document.file.path, 'rb') as file:
            response.write(file.read())
        
        return response

# TODO:
class ReportDownloadView(DetailView):
    def get(self, request, *args, **kwargs):
        pdd = get_object_or_404(PDD, id=kwargs["report_id"], active=True)
        context = {"pdd": pdd}
        
        # check if the file exists
        if not os.path.exists(pdd.pdd.file.path):
            return HttpResponse(_("File does not exist"), status=404)

        # set the file content type and headers
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{pdd.pdd.name}"'
        
        # read the file and write it to the response
        with open(pdd.pdd.file.path, 'rb') as file:
            response.write(file.read())
        
        return response

# TODO:
class ReportListView(ListView):
    model = PDD
    template_name = "reporting/list.html"
    paginate_by = 50

    def get_context_data(self, **kwargs):
        context = super(ReportListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Filtra los objetos por case_study_id
        queryset = PDD.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')
        
        return queryset

#############################################################################################
#############################################################################################
#############################################################################################
## VERSIONES DE ALE

# class ReportGenerateView_ALE(DetailView):
#     def get(self, request, *args, **kwargs):
#         model = Execution
#         execution = get_object_or_404(Execution, id=kwargs["execution_id"]) 

#         pdd = PDD.objects.create(
#             execution=execution,
#             user=request.user  
#         )
#         report_path = os.path.join(execution.exp_folder_complete_path,'exec_'+str(execution.id)+'_reports', 'report.docx')

#         # Simulate report creation (you should replace this with your actual report generation logic)
#         with open(report_path, 'wb') as file:
#             file.write(b'PDF content or whatever content you generate')

#         pdd.file.name = report_path
#         pdd.save()

#         return response
    

#########################################################################


def tree_to_png(path_to_tree_file):
    # Cargar los datos del árbol de decisión
    try:
        with open('/screenrpa/' + path_to_tree_file, 'rb') as archivo:
            loaded_data = pickle.load(archivo)
        clasificador_loaded = loaded_data['classifier']
        feature_names_loaded = loaded_data['feature_names']
        class_names_loaded = [str(item) for item in loaded_data['class_names']]
    except FileNotFoundError:
        print(f"File not found: {path_to_tree_file}")
        return None
    
    dot_data = io.StringIO()
    export_graphviz(clasificador_loaded, out_file=dot_data, filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names_loaded, class_names=class_names_loaded)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    png_image = graph.create_png()

    # Guardar la imagen en un archivo temporal
    temp_file = NamedTemporaryFile(delete=False, suffix='.png')
    with open(temp_file.name, 'wb') as f:
        f.write(png_image)
    
    return temp_file.name

#############################################################################################
class ReportCreateView(CreateView):
    model = PDD
    form_class = ReportingForm
    template_name = "reporting/create.html"

    def get_context_data(self, **kwargs):
        context = super(ReportCreateView, self).get_context_data(**kwargs)
        
        execution_id = self.kwargs.get('execution_id')
        execution = Execution.objects.get(pk=execution_id)
        context['execution'] = execution
        reports = PDD.objects.filter(execution=execution).order_by('-created_at')
        context['reports'] = reports
        return context 
    
    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError(_("User must be authenticated."))
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.execution = Execution.objects.get(pk=self.kwargs.get('execution_id'))
        saved = self.object.save()

        self.report_generate(self.object)

        return HttpResponseRedirect(self.get_success_url())
    
    def report_generate(self, report):
        execution= self.get_context_data()['execution']
        scenarios_to_study= self.get_context_data()['execution'].scenarios_to_study

        for scenario in scenarios_to_study:

            report_directory = os.path.join(execution.exp_folder_complete_path, scenario+"_results")
            report_path = os.path.join(report_directory, f'report_{report.id}.docx')

            # Ensure the directory exists
            os.makedirs(report_directory, exist_ok=True)

            report_define(report_directory, report_path, execution, report, scenario)
            # with open(report_path, 'wb') as file:
            #     file.write(b'PDF content or whatever content you generate')
        
#############################################################################################


def report_define(report_directory, report_path, execution,  report, scenario):
    template_path = "/screenrpa/media/unzipped/report_template.docx"
    doc = Document(template_path)

    ###############3
    paragraph_dict = {}
    for i, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text
        if text.startswith('[') and text.endswith(']'): 
            paragraph_dict[text] = i 

    for key, value in paragraph_dict.items():
        print(f"Texto del párrafo: {key}\nÍndice: {value}\n")

    ############################
    
    purpose= doc.paragraphs[paragraph_dict['[PURPOSE]']]
    purpose.text = report.purpose
    #paragraph.style = doc.styles['Normal']

    purpose= doc.paragraphs[paragraph_dict['[OBJECTIVE]']]
    purpose.text = report.objective

    title= doc.paragraphs[paragraph_dict['[TITLE]']]
    title.text = execution.process_discovery.title

    #falta [SHORT DESCRIPTION] --> meter el campo en process discovery
    ############################

    nameapps= doc.paragraphs[paragraph_dict['[DIFERENT NAMEAPPS]']]
    #nameapps.style = doc.styles['ListBullet'] --> add_paragraph('text', style='ListBullet')

    logcsv_directory = os.path.join(execution.exp_folder_complete_path,scenario,'log.csv')

    df_logcsv = pd.read_csv(logcsv_directory)

    unique_names = df_logcsv['NameApp'].unique()

    for name in unique_names:
        run = nameapps.add_run('\n• ' + name)
        run.add_break()
        row = df_logcsv[df_logcsv['NameApp'] == name].iloc[0]
        screenshot_filename = row['Screenshot']
        screenshot_directory = os.path.join(execution.exp_folder_complete_path, scenario, screenshot_filename)
        # Insertar la imagen si existe
        if os.path.exists(screenshot_directory):
        # Agregar la imagen debajo del nombre
            nameapps.add_run().add_picture(screenshot_directory, width=Inches(6))
        else:
            print(f"Image not found for {name}: {screenshot_directory}")

    ##########################3

    original_log= doc.paragraphs[paragraph_dict['[ORIGINAL LOG]']]

    table = doc.add_table(rows=(df_logcsv.shape[0] + 1), cols=df_logcsv.shape[1])

    # Insertar los nombres de las columnas
    for j, col in enumerate(df_logcsv.columns):
        table.cell(0, j).text = col

    # Insertar los datos del DataFrame
    for i in range(df_logcsv.shape[0]):
        for j in range(df_logcsv.shape[1]):
            table.cell(i + 1, j).text = str(df_logcsv.iloc[i, j])

    tbl, p = table._tbl, original_log._p
    p.addnext(tbl)
    ###################33333
    #meter el decision tree --> document.add_picture('image_path') : agrega una imagen al documento.

    decision_tree= doc.paragraphs[paragraph_dict['[DECISION TREE]']]

    path_to_tree_file = os.path.join(execution.exp_folder_complete_path, scenario+"_results", "decision_tree_ale.pkl")
    
    run = decision_tree.add_run()
    run.add_picture(tree_to_png(path_to_tree_file), width=Inches(6))

    #decision_tree.text = ''

    ####################
    doc.save(report_path)


#####################################################################

def deleteReport(request):
    report_id = request.GET.get("report_id")

    removed_report = PDD.objects.get(id=report_id)
    if request.user.id != removed_report.user.id:
        raise Exception(_("This case study doesn't belong to the authenticated user"))
    #podria ser interesante, para no borrar un reporte que se esta generando

    #if removed_report.executed != 0:
    #    raise Exception(_("This case study cannot be deleted because it has already been excecuted"))

    execution = removed_report.execution

    scenarios_to_study = execution.scenarios_to_study


    for scenario in scenarios_to_study:
        report_directory = os.path.join(execution.exp_folder_complete_path, scenario+"_results")
        report_path = os.path.join(report_directory, f'report_{report_id}.docx')
        if os.path.exists(report_path):
            os.remove(report_path)

    removed_report.delete()

    return HttpResponseRedirect(reverse("analyzer:execution_detail", kwargs={"execution_id": removed_report.execution.id}))

#############################################################################################3

# def reportingConfigurationDetail(request):
    
#     report = get_object_or_404(PDD, pk=report_id)

#     form = ReportingForm(read_only=True, instance=report)  # Todos los campos estarán desactivados

#     return render(request, 'reporting/create.html', {'form': form})


class ReportingConfigurationDetail(DetailView):
    def get(self, request, *args, **kwargs):
        report = get_object_or_404(PDD, id=kwargs["report_id"])
        
        form = ReportingForm(read_only=True, instance=report)  # Todos los campos estarán desactivados
        context = {"form": form,
            "execution": report.execution,
            }
        return render(request, 'reporting/detail.html', context)
    

##################################################3

def download_report_zip(request, report_id):
    report = get_object_or_404(PDD, pk=report_id)
    execution= report.execution

    zip_filename = f"execution_{execution.id}_reports_{report.id}.zip"

    zip_path = os.path.join(settings.MEDIA_ROOT, zip_filename)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        scenarios_to_study = execution.scenarios_to_study

        for scenario in scenarios_to_study:

            report_directory = os.path.join(execution.exp_folder_complete_path, scenario + "_results")
            report_filename = f'report_{report.id}.docx'
            report_path = os.path.join(report_directory, report_filename)
            
            new_filename = f'scenario_{scenario}_report_{report.id}.docx'

            # Ensure the file exists before adding to ZIP
            if os.path.exists(report_path):
                zipf.write(report_path, arcname=new_filename)

    with open(zip_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/zip')
        response['Content-Disposition'] = f'attachment; filename={zip_filename}'

    os.remove(zip_path)  # Clean up the generated zip file after serving it
    return response