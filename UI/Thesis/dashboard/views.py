from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
import paramiko
from subprocess import Popen, PIPE
import time
from django.http import JsonResponse
from django.shortcuts import redirect

import pprint
import re

code_gen = ""

def index(request):

    # return HttpResponse("Hello, world. You're at the polls index.")
    return render(request, 'dashboard/index.html')


def temp(request):
    return HttpResponse("Ajax done!")

def execute_ssh(cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    server = ''
    username = ''
    password = ''

    ssh.connect(server, username=username, password=password)
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd)
    time.sleep(7)
    print('something')
    stdout = ssh_stdout.readlines()
    ssh.close()

    return stdout

#Check for function present
def check_function(expression):
    open_tup = tuple('({[')
    close_tup = tuple(')}]')
    map = dict(zip(open_tup, close_tup))
    queue = []

    for i in expression:
        if i in open_tup:
            queue.append(map[i])
        elif i in close_tup:
            if not queue or i != queue.pop():
                return "Unbalanced"
    if not queue:
        return "Balanced"
    else:
        return "Unbalanced"

def number_of_functions(txt):
    ans = check_function(txt)
    if ans == 'Balanced':
        return txt.count('(')
    else:
        return 0

def get_func_template(var_name):
    template_code = '''\t def %s ():
            pass\n''' % var_name
    return template_code


def structure_check(struct_str, element_bool):
    if element_bool == "return":
        try:
            num_of_func = number_of_functions(struct_str)
            if num_of_func >= 1:
                final_str_code = ''
                #for checking functions
                # pattern = re.compile(".*?\((.*?)\)")
                # l = pattern.match(struct_str)
                ans = check_function(struct_str)
                if ans == 'Balanced':
                    # num_of_func = number_of_functions(struct_str)
                    occurence = [_.start() for _ in re.finditer("\(", struct_str)]
                    for occ in occurence:
                        var_name = struct_str[:occ].rstrip().split(' ')[-1]
                        if struct_str.count(var_name) <= 1:
                            final_str_code = final_str_code + get_func_template(var_name).lstrip()

                    # variable_function_name = struct_str.split('(')[0].rstrip().split(' ')[-1]

                    template_return_stmt = final_str_code + '\n\t' + struct_str.lstrip().rstrip()
                    return template_return_stmt

                if ans == 'Unbalanced':
                    print('Error Generation! Improper Generation of Function Detected!')

            else:
                print('No Function')
                return struct_str.lstrip().rstrip()

        except Exception as e:
            print(e)


def get_for_loop_template(txt):
    if txt.count('for') > 1:
        var_name = txt.split('in')[1].lstrip().split(' ')[0]
        if txt.count(var_name) <= 1:
            template_for = '''%s = []\n''' % var_name
            rslt = template_for + txt
        return rslt
    else:
        return txt

def ssh_model(request):
    ccode = request.GET.get('annotation_method', '')
    ccode_body = request.GET.get('annotation_body', '')

    #FOR METHOD SIGN
    test_code = '''<annotation_start_m> ''' + ccode + ''' <annotation_end_m>'''
    test_code_body = '''<annotation_start_b> ''' + ccode_body +''' <annotation_end_b>'''


    # cmd_to_execute = 'bash /home/mushir/final/remotescript.sh "%s" %s' % (test_code, len_gen)
    final = ''
    try:
        if ccode != '':
            cmd_to_execute = 'conda activate new_venv && python /home/mushir/final/eval.py "%s"' % test_code
            f = execute_ssh(cmd_to_execute)[0].split("<method_sign_end>")[0].split("<method_sign_start>")[1].lstrip().rstrip().replace(":",":\n")
            final = final + f

        if ccode_body != '':
            cmd_to_execute = 'conda activate new_venv && python /home/mushir/final/eval.py "%s"' % test_code_body
            body_code = str(execute_ssh(cmd_to_execute))
            b = body_code.split('<annotation_end_b>#<body_start>')
            b2 = b[1].split('<body_end>')[0]
            # b2 = '''            for node in self. nodelist :               try :                    if isinstance ( node, Node ) :                       bit = self. render_node ( node, context )  else :                      bit = node  bits. append ( force_text ( bit ) )  return mark_safe ( ''. join ( bits ) ) '''
            b2 = b2.lstrip()
            b2 = b2.replace('self.', '')
            b2 = b2.replace('_','')
            b2 = b2.replace(":", ":\n")
            if b2.count('else') >= 1:
                b2 = b2.split('else')[0]


            find_return = b2.find('return')
            find_for = b2.find('for ')
            find_if = b2.find('if ')
            is_find_for = False
            is_find_if = False
            is_return_start = False
            rslt = ''

            if find_return == 0:
                is_return_start = True
                rslt = structure_check(b2, "return")
            else:
                tobeinserted = ''
                get_num_func = number_of_functions(b2)
                if get_num_func >= 1:
                    occurence = [_.start() for _ in re.finditer("\(", b2)]
                    for occ in occurence:
                        var_name = b2[:occ].rstrip().split(' ')[-1]
                        if b2.count(var_name) <= 1:
                            tobeinserted = tobeinserted + get_func_template(var_name).lstrip()

                rslt = tobeinserted + '\n\t' + b2.lstrip().rstrip()

                rslt = get_for_loop_template(rslt)

            final = final + rslt

        ##Making sense of the code

    except Exception as e:
        print(e)

    global code_gen
    code_gen = final
    # return redirect('display_results')

    return JsonResponse({'text': final})

def display_results(request):
    # code_gen = request.GET.get('data', '')
    global code_gen

    return HttpResponse(code_gen)

def ssh_model_body(request):
    ccode = request.GET.get('annotation_method', '')
    #FOR METHOD SIGN
    test_code = '''<annotation_start_m> ''' + ccode + ''' <annotation_end_m>'''

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    server = 'elwe1.rhrk.uni-kl.de'
    username = 'mushir'
    password = '4778367530ah@d'
    # cmd_to_execute = 'bash /home/mushir/final/remotescript.sh "%s" %s' % (test_code, len_gen)
    cmd_to_execute = 'conda activate new_venv && python /home/mushir/final/eval.py "%s"' % test_code
    try:
        ssh.connect(server, username=username, password=password)
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd_to_execute)
        time.sleep(15)
        print('something')
        stdout = ssh_stdout.readlines()
        ssh.close()
    except Exception as e:
        print(e)

    return HttpResponse(stdout)



def other_ssh():
    stdout, stderr = Popen(['ssh', 'user@remote_computer', 'ps -ef'], stdout = PIPE).communicate()
    print(stdout)
    return 1

# def sub_proc():
#     subprocess.Popen(f"ssh {user}@{host} {cmd}", shell=True, stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE).communicate()